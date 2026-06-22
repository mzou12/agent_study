import 'dotenv/config';
import { MultiServerMCPClient } from '@langchain/mcp-adapters';
import { ChatOpenAI } from '@langchain/openai';
import chalk from 'chalk';
import { HumanMessage, SystemMessage, ToolMessage } from '@langchain/core/messages';

/**
 * 1. MCP Client 根据配置准备连接 MCP Server
 * 2. MCP Server启动并注册工具和资源
 * 3. MCP Client连接 MCP Server
 * 4. MCP Client获取工具和资源
 * 5. MCP Client将tool转化为 tool calling格式传输入LLM
 * 6. LLM看到工具和问题
 * 7. LLM决定调用工具
 * 8. LLM输出tool call
 * 9. MCP Client收到tool call，向MCP Server通过特定接口发出tools/ call请求
 * 10. MCP Server收到tools/ call请求，在本地执行绑定的tool
 * 11. MCP Server将tool结果返回给MCP Client
 * 12. MCP Client收到tool结果，将tool结果包装成 tool message 加入 messages，然后再次发给LLM
 * 13. LLM看到tool结果，决定是否继续调用工具
 * 14. LLM输出最终结果
 * 15. MCP Client收到最终结果，返回给用户
 * 16. MCP Client关闭连接
 */
const model = new ChatOpenAI({ 
    modelName: "qwen-plus",
    apiKey: process.env.OPENAI_API_KEY,
    configuration: {
        baseURL: process.env.OPENAI_BASE_URL,
    },
});

const mcpClient = new MultiServerMCPClient({
    mcpServers: {
        'my-mcp-server': {
            command: "node",
            args: [
                "/home/zou/agent_study/tool-test/src/my-mcp-server.mjs"
            ]
        }
    }
});

const tools = await mcpClient.getTools();
const modelWithTools = model.bindTools(tools);

const res = await mcpClient.listResources();

let resourceContent = '';
for (const [serverName, resources] of Object.entries(res)) {
    for (const resource of resources) {
        const content = await mcpClient.readResource(serverName, resource.uri);
        resourceContent += content[0].text;
    }
}

async function runAgentWithTools(query, maxIterations = 30) {
    const messages = [
        new SystemMessage(resourceContent),
        new HumanMessage(query)
    ];

    for (let i = 0; i < maxIterations; i++) {
        console.log(chalk.bgGreen(`⏳ 正在等待 AI 思考...`));
        const response = await modelWithTools.invoke(messages);
        messages.push(response);

        // 检查是否有工具调用
        if (!response.tool_calls || response.tool_calls.length === 0) {
            console.log(`\n✨ AI 最终回复:\n${response.content}\n`);
            return response.content;
        }

        console.log(chalk.bgBlue(`🔍 检测到 ${response.tool_calls.length} 个工具调用`));
        console.log(chalk.bgBlue(`🔍 工具调用: ${response.tool_calls.map(t => t.name).join(', ')}`));
        // 执行工具调用
        for (const toolCall of response.tool_calls) {
            const foundTool = tools.find(t => t.name === toolCall.name);
            if (foundTool) {
                const toolResult = await foundTool.invoke(toolCall.args);
                messages.push(new ToolMessage({
                    content: toolResult,
                    tool_call_id: toolCall.id,
                }));
            }
        }
    }

    return messages[messages.length - 1].content;
}


await runAgentWithTools("查一下用户 002 的信息");
// await runAgentWithTools("MCP Server 的使用指南是什么");

await mcpClient.close();