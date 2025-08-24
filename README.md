# LangGraph Workflow 项目

一个基于 [LangGraph](https://github.com/langchain-ai/langgraph) 构建的高级工作流系统，实现了复杂的多步骤任务处理和状态管理。

![LangGraph Workflow 架构图]
<img width="552" height="1172" alt="wf" src="https://github.com/user-attachments/assets/78e6890c-25c9-4f3d-80b5-e1a2ecf1e3b3" />

## 架构概述

本项目采用 LangGraph 的图形化工作流设计，将复杂任务分解为多个可管理的节点，并通过有向边连接这些节点，形成一个完整的工作流执行管道。

### 核心组件

- **节点设计**: 每个节点代表一个特定的处理单元，可以是 LLM 调用、工具使用或自定义逻辑
- **条件边**: 支持基于当前状态的动态路由决策

## 视频演示

点击下方视频查看工作流运行演示：

[![LangGraph Workflow 演示视频]


https://github.com/user-attachments/assets/9060d97d-501b-4419-9600-5e096d02a5c9




## 项目结构

```
project/
├── src/                    # 源代码
│   └── agent
│       └── graphs.py            # 工作流图定义
```


## 贡献指南

我们欢迎社区贡献！请参阅 [CONTRIBUTING.md](./CONTRIBUTING.md) 了解如何参与项目开发。

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 支持

如果您遇到问题或有疑问：

1. 查看 [文档](./README.md)
2. 提交 [GitHub Issue](https://github.com/DHU-JinQi/Langgraph_workflow_architecture/issues)



*此项目基于 [LangGraph](https://github.com/langchain-ai/langgraph) 构建，LangGraph 是 LangChain 生态系统的一部分。*
