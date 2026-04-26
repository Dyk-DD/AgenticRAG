"""
基于图RAG的临床决策与辅助系统 - 主程序
整合混合检索和图RAG检索，实现临床决策辅助和医学图谱数据推理
"""

import os
import sys
import time
import json
import logging
from typing import List, Optional

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, GraphRAGConfig

# 导入重构后的临床医学模块
from rag_modules.graph_data_preparation import MedicalDataPreparationModule
from rag_modules.milvus_index_construction import MilvusIndexConstructionModule
from rag_modules.generation_integration import GenerationIntegrationModule
from rag_modules.hybrid_retrieval import HybridRetrievalModule
from rag_modules.graph_rag_retrieval import GraphRAGRetrieval
from rag_modules.intelligent_query_router import IntelligentQueryRouter

# 加载环境变量
load_dotenv()

class ClinicalDecisionSystem:
    """
    临床决策与辅助系统
    核心特性：
    1. 智能路由：自动评估病情查询复杂度，精准分发至最优检索引擎
    2. 双引擎检索：传统混合检索(基础医学知识) + 图RAG检索(复杂禁忌与鉴别诊断)
    3. 临床图谱推理：多跳图遍历、临床路径发现、并发症风险推理
    """
    
    def __init__(self, config: Optional[GraphRAGConfig] = None):
        self.config = config or DEFAULT_CONFIG

        # 初始化大模型客户端
        from openai import OpenAI
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            logger.warning("未检测到 DEEPSEEK_API_KEY 环境变量，请确保已配置")

        self.llm_client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        
        # 核心模块
        self.data_module = None
        self.index_module = None
        self.generation_module = None
        
        # 检索引擎
        self.traditional_retrieval = None
        self.graph_rag_retrieval = None
        self.router = None
        
        # 系统状态
        self.system_ready = False

    def initialize_system(self):
        """初始化系统所有组件"""
        logger.info("开始初始化临床决策系统组件...")

        # 1. 医疗数据准备模块 - 新增传入 config 和 llm_client
        self.data_module = MedicalDataPreparationModule(
            config=self.config,           # <-- 新增
            csv_dir=self.config.medical_csv_dir,
            uri=self.config.neo4j_uri,
            user=self.config.neo4j_user,
            password=self.config.neo4j_password
        )

        # 2. Milvus 向量索引构建模块
        self.index_module = MilvusIndexConstructionModule(
            host=self.config.milvus_host,
            port=self.config.milvus_port,
            collection_name="clinical_qa_knowledge",  # 确保与之前重构的一致
            model_name=self.config.embedding_model
        )

        # 3. 临床文本生成集成模块
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=0.1  # 医疗场景保持低温度以防幻觉
        )

        # 4. 检索模块实例化
        self.traditional_retrieval = HybridRetrievalModule(
            config=self.config,
            milvus_module=self.index_module,
            data_module=self.data_module,
            llm_client=self.llm_client
        )

        self.graph_rag_retrieval = GraphRAGRetrieval(
            config=self.config,
            llm_client=self.llm_client
        )

        # 5. 智能医疗查询路由器
        self.router = IntelligentQueryRouter(
            traditional_retrieval=self.traditional_retrieval,
            graph_rag_retrieval=self.graph_rag_retrieval,
            llm_client=self.llm_client,
            config=self.config
        )

        logger.info("系统组件初始化完成")

    def build_knowledge_base(self):
        """构建临床知识库：解析CSV、构建图谱与向量索引"""
        logger.info("开始构建临床知识库...")

        # 1. 加载 CSV 医疗问答数据
        self.data_module.load_csv_data()

        # 2. 检查并构建 Neo4j 图谱
        if self.data_module.check_graph_needs_update():  # 改用新的方法名
            logger.info("开始执行图谱写入/增量更新...")
            self.data_module.build_neo4j_graph()
        else:
            logger.info("✅ 图数据库 (Neo4j) 数据已同步。跳过耗时的写入步骤。")

        # 3. 构建医疗文档与块 (按照一条QA一个Chunk的逻辑)
        docs = self.data_module.build_medical_documents()
        chunks = self.data_module.chunk_documents()

        # 4. 构建或加载 Milvus 向量索引
        if not self.index_module.has_collection():
            logger.info("向量集合不存在，开始构建...")
            self.index_module.build_vector_index(chunks)
        else:
            logger.info("向量集合已存在，直接加载...")
            self.index_module.load_collection()

        # 5. 初始化检索器底层数据结构 (包括加载Neo4j图谱)
        self.traditional_retrieval.initialize(chunks)
        self.graph_rag_retrieval.initialize()

        # 👇 就是补上下面这一行代码 👇
        self.system_ready = True

        logger.info("临床知识库构建与就绪完成！")

    def _initialize_retrievers(self, chunks: List = None):
        """初始化临床检索引擎"""
        print("初始化临床检索引擎...")

        # 如果没有chunks，从数据模块获取
        if chunks is None:
            chunks = self.data_module.chunks or []

        # 初始化传统检索器
        self.traditional_retrieval.initialize(chunks)

        # 初始化图RAG检索器
        self.graph_rag_retrieval.initialize()

        self.system_ready = True
        print("✅ 临床检索引擎初始化完成！")

    def _show_knowledge_base_stats(self):
        """显示临床知识库统计信息"""
        print(f"\n🏥 临床知识库统计:")

        # 数据统计 (适配 MedicalDataPreparationModule 的返回结构)
        stats = self.data_module.get_statistics()
        print(f"   医疗问答对: {stats.get('total_qa_pairs', 0)} 条")
        print(f"   涉及科室数: {stats.get('total_departments', 0)} 个")
        print(f"   文档总数量: {stats.get('total_documents', 0)} 份")
        print(f"   文本块数量: {stats.get('total_chunks', 0)} 块")

        # Milvus统计
        milvus_stats = self.index_module.get_collection_stats()
        print(f"   向量索引库: {milvus_stats.get('row_count', 0)} 条记录")

        # 图RAG路由统计
        # 注意：这里的属性名根据你 init 里的定义，如果叫 self.router 就用 self.router
        router = getattr(self, 'router', getattr(self, 'query_router', None))
        if router:
            route_stats = router.get_route_statistics()
            print(f"   智能路由网: 累计处理问诊 {route_stats.get('total_queries', 0)} 次")

        # 打印核心科室分布
        if stats.get('departments_distribution'):
            departments = list(stats['departments_distribution'].keys())[:10]
            print(f"   🏷️ 核心科室: {', '.join(departments)}")

    def ask_question_with_routing(self, question: str, stream: bool = False, explain_routing: bool = False):
        """
        智能临床问答：自动选择最佳检索与推演策略
        """
        if not getattr(self, 'system_ready', False):
            raise ValueError("系统未就绪，请先构建临床知识库")

        print(f"\n🩺 患者/医生提问: {question}")

        router = getattr(self, 'router', getattr(self, 'query_router', None))

        # 显示路由决策解释（可选）
        if explain_routing and router:
            explanation = router.explain_routing_decision(question)
            print(explanation)

        start_time = time.time()

        try:
            # 1. 智能路由检索
            print("执行智能临床查询路由...")
            relevant_docs, analysis = router.route_query(question, self.config.top_k)

            # 2. 显示路由信息
            strategy_icons = {
                "hybrid_traditional": "🔍",
                "graph_rag": "🧬",
                "combined": "🏥"
            }
            strategy_icon = strategy_icons.get(analysis.recommended_strategy.value, "❓")
            print(f"{strategy_icon} 系统调度策略: {analysis.recommended_strategy.value}")
            print(f"📊 临床复杂度: {analysis.query_complexity:.2f}, 关系密集度: {analysis.relationship_intensity:.2f}")

            # 3. 显示检索结果信息
            if relevant_docs:
                doc_info = []
                for doc in relevant_docs:
                    # 替换烹饪的 recipe_name 为医学的 title / department
                    title = doc.metadata.get('title', '未知病例')
                    search_type = doc.metadata.get('search_type', doc.metadata.get('route_strategy', 'unknown'))
                    score = doc.metadata.get('final_score', doc.metadata.get('relevance_score', 0))
                    doc_info.append(f"《{title}》({search_type}, {score:.3f})")

                print(f"📋 找到 {len(relevant_docs)} 份相关临床参考: {', '.join(doc_info[:3])}")
                if len(doc_info) > 3:
                    print(f"    等共 {len(relevant_docs)} 个参考结果...")
            else:
                return "抱歉，没有找到相关的临床医学参考信息。请尝试补充更多症状或更换提问方式。", analysis

            # 4. 生成回答
            print("👨‍⚕️ 智能生成临床辅助建议...\n")

            if stream:
                try:
                    for chunk_text in self.generation_module.generate_adaptive_answer_stream(question, relevant_docs):
                        print(chunk_text, end="", flush=True)
                    print("\n")
                    result = "流式输出完成"
                except Exception as stream_error:
                    logger.error(f"流式输出过程中出现错误: {stream_error}")
                    print(f"\n⚠️ 网络或流式输出中断，正在为您切换到标准模式...")
                    result = self.generation_module.generate_adaptive_answer(question, relevant_docs)
                    print(result)
            else:
                result = self.generation_module.generate_adaptive_answer(question, relevant_docs)

            # 5. 性能统计
            end_time = time.time()
            print(f"\n⏱️ 诊断处理完成，耗时: {end_time - start_time:.2f}秒")

            return result, analysis

        except Exception as e:
            logger.error(f"临床问答处理失败: {e}")
            return f"抱歉，处理医疗请求时出现系统错误：{str(e)}", None

    def run_interactive(self):
        """运行交互式问诊循环"""
        if not getattr(self, 'system_ready', False):
            print("❌ 系统未就绪，请先构建临床知识库")
            return

        print("\n" + "=" * 55)
        print("欢迎使用【临床决策与辅助决策系统】！")
        print("可用功能指令：")
        print("   - 'stats'   : 查看系统当前挂载数据与路由统计")
        print("   - 'rebuild' : 重置并重建临床知识图谱与向量库")
        print("   - 'add <路径>': 增量导入新的CSV数据 (如: add data/new_qa.csv)")  # <-- 新增菜单提示
        print("   - 'quit'    : 安全退出系统")
        print("=" * 55)

        while True:
            try:
                user_input = input("\n🩺 请输入医学咨询问题: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'q', 'exit']:
                    break
                elif user_input.lower() == 'stats':
                    self._show_system_stats()
                    continue
                elif user_input.lower() == 'rebuild':
                    self._rebuild_knowledge_base()
                    continue
                # 👇 新增指令拦截：处理增量添加
                elif user_input.lower().startswith('add '):
                    csv_path = user_input[4:].strip()
                    self.add_new_knowledge(csv_path)
                    continue

                use_stream = True  # 默认流式输出
                explain_routing = False  # 如果需要看路由推理过程可以设为 True

                result, analysis = self.ask_question_with_routing(
                    user_input,
                    stream=use_stream,
                    explain_routing=explain_routing
                )

                if not use_stream and result:
                    print(f"{result}\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")
                import traceback
                traceback.print_exc()

        print("\n👋 感谢使用临床决策辅助系统，再见！")
        self._cleanup()

    def _show_system_stats(self):
        """显示临床系统统计信息"""
        print("\n🏥 临床系统运行统计")
        print("=" * 40)

        router = getattr(self, 'router', getattr(self, 'query_router', None))
        if router:
            route_stats = router.get_route_statistics()
            total_queries = route_stats.get('total_queries', 0)

            if total_queries > 0:
                print(f"累计响应问诊: {total_queries} 次")
                print(
                    f"传统检索 (基础查询): {route_stats.get('traditional_count', 0)} ({route_stats.get('traditional_ratio', 0):.1%})")
                print(
                    f"图RAG检索 (复杂推理): {route_stats.get('graph_rag_count', 0)} ({route_stats.get('graph_rag_ratio', 0):.1%})")
                print(
                    f"组合策略 (疑难杂症): {route_stats.get('combined_count', 0)} ({route_stats.get('combined_ratio', 0):.1%})")
            else:
                print("暂无临床问诊记录")

        self._show_knowledge_base_stats()

    def _rebuild_knowledge_base(self):
        """重建临床知识库"""
        print("\n⚠️ 准备重建临床知识库...")

        confirm = input("⚠️  这将清空现有的医疗向量数据并重新解析重建，是否继续？(y/N): ").strip().lower()
        if confirm != 'y':
            print("❌ 重建操作已取消")
            return

        try:
            print("清理现有的 Milvus 医疗向量集合...")
            if self.index_module.delete_collection():
                print("✅ 现有数据集合已清除")
            else:
                print("清理集合时出现警告，尝试强制覆盖重建...")

            print("开始重新构建临床知识图谱与向量库...")
            self.build_knowledge_base()

            print("✅ 临床知识库重建完成，系统已就绪！")

        except Exception as e:
            logger.error(f"重建临床知识库失败: {e}")
            print(f"❌ 重建失败: {e}")
            print("建议：请检查 Milvus 和 Neo4j 服务连接状态后重试")
    
    def _cleanup(self):
        """清理资源"""
        if getattr(self, 'data_module', None):
            # 适配可能没有 close 方法的模块
            if hasattr(self.data_module, 'close'): self.data_module.close()
        if getattr(self, 'traditional_retrieval', None):
            self.traditional_retrieval.close()
        if getattr(self, 'graph_rag_retrieval', None):
            self.graph_rag_retrieval.close()
        if getattr(self, 'index_module', None):
            self.index_module.close()

    def add_new_knowledge(self, csv_filepath: str):
        """增量添加新知识：处理全新的CSV文件并追加到数据库"""
        if not os.path.exists(csv_filepath):
            print(f"❌ 找不到文件: {csv_filepath}")
            return

        print(f"\n📦 开始增量导入新数据: {csv_filepath}")

        # 1. 读取新的 CSV 数据
        self.data_module.load_single_csv(csv_filepath)
        if not getattr(self.data_module, 'qa_pairs', None):
            print("❌ 没有提取到有效数据，增量导入终止")
            return

        # 2. 追加到 Neo4j 图数据库
        # 注意：由于我们在 Cypher 中使用了 MERGE 且具有了全局唯一哈希 ID，
        # 重复数据会被自动忽略，新数据会被平滑追加，不会发生冲突。
        print("正在将新实体和关系写入图谱引擎 (Neo4j)...")
        self.data_module.build_neo4j_graph()

        # 3. 构建新的文档与文本块
        print("正在处理新的文档和向量块...")
        self.data_module.build_medical_documents()
        new_chunks = self.data_module.chunk_documents()

        # 4. 追加到 Milvus 向量数据库
        if new_chunks:
            print(f"正在将 {len(new_chunks)} 条新向量追加到 Milvus 数据库...")
            if self.index_module.add_documents(new_chunks):
                print("✅ 向量库追加成功！")
            else:
                print("❌ 向量库追加失败！")

        # 5. 刷新系统的内存检索引擎状态
        print("正在刷新混合检索引擎状态，加载最新数据...")
        if getattr(self, 'traditional_retrieval', None):
            # 重置图检索模块的标记，使其在下次检索时从 Neo4j 重新拉取最新图谱结构
            self.traditional_retrieval.graph_indexed = False
            self.traditional_retrieval._build_graph_index()

            # 将新的文本块追加到 BM25 检索器中
            if self.traditional_retrieval.bm25_retriever:
                self.traditional_retrieval.bm25_retriever.add_documents(new_chunks)

        print("🎉 增量知识更新完成！您可以立即开始针对新数据的问诊。")


def main():
    """主函数入口"""
    try:
        print("🚀 正在启动 临床决策与辅助系统 (Clinical Decision Support System)...")

        # 使用你新定义的类名
        rag_system = ClinicalDecisionSystem()

        # 初始化系统
        rag_system.initialize_system()

        # 构建知识库
        rag_system.build_knowledge_base()

        # 运行交互式问诊循环
        rag_system.run_interactive()

    except Exception as e:
        logger.error(f"系统运行失败: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n❌ 系统致命错误: {e}")


if __name__ == "__main__":
    main() 