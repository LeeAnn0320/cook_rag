"""
主程序
"""

import os
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent))
from typing import List

from dotenv import load_dotenv
from config import DEFAULT_CONFIG,RAGConfig

from rag_modules import(
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule
)

#加载环境变量
load_dotenv()

import logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger=logging.getLogger(__name__)


class RecipeRAGSystem:
    def __init__(self,config:RAGConfig=None):
        self.config=config or DEFAULT_CONFIG
        self.data_module=None
        self.index_module=None
        self.retrieval_module=None
        self.generation_module=None
        #检查数据路径
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在:{self.config.data_path}")

        #检查API
        if not os.getenv("DASHSCOPE_API_KEY"):
            raise ValueError("请设置API")
    def init_system(self):
        """初始化所有模块"""
        logger.info(f"正在初始化模块")
        self.data_module=DataPreparationModule(self.config.data_path)       
        logger.info(f"数据模块初始化完成")
        self.index_module=IndexConstructionModule(model_name=self.config.embedding_model,index_save_path=self.config.index_save_path)
        logger.info(f"索引模块初始化完成")
        self.generation_module=GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        logger.info(f"生成模块初始化完成")
    def build_knowledge(self):
        logger.info("正在初始化知识库")
        #1.尝试加载已保存的索引
        vectorstore=self.index_module.load_index()
        if vectorstore is not None:
            logger.info("成功加载已保存的向量索引")
            self.data_module.load_document()
            logger.info("加载食谱文档")
            chunks=self.data_module.chunk_documents()
            logger.info("进行文本分块中")
        else:
            logger.info("未找到保存的向量索引，开始构建新索引")
            logger.info("加载食谱文档")
            self.data_module.load_document()
            logger.info("进行文本分块中")
            chunks=self.data_module.chunk_documents()
            logger.info("开始构建向量索引")
            self.index_module.build_vector_index(chunks)
            logger.info("保存向量索引")
            self.index_module.save_index()

        logger.info("初始化检索")
        self.retrieval_module=RetrievalOptimizationModule(vectorstore,chunks)
        #显示统计信息
        stats=self.data_module.get_statics()
        print(f"\n📊 知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print(f"   菜品分类: {list(stats['categories'].keys())}")
        print(f"   难度分布: {stats['difficulties']}")
        logger.info("知识库构建完成")

    def ask_question(self,question:str,stream:bool=False):
        """
        回答用户问题

        Args:
            question: 用户问题
            stream: 是否使用流式输出

        Returns:
            生成的回答或生成器
        """
        if not all([self.retrieval_module,self.generation_module]):
            raise ValueError("请先构建知识库")
        print(f"用户问题:{question}")
        #1.查询路由
        router_type=self.generation_module.query_router(query=question)
        print(f"查询类型:{router_type}")

        #2.智能查询重写
        if router_type =="list":
            rewritten_query=question
            print(f"列表查询保持原样:{rewritten_query}")
        else:
            print("智能分析查询")
            rewritten_query=self.generation_module.query_rewrite(query=question)

        #3.检索相关子块
        print("检索相关文档")
        filters=self._extract_filters_from_query(query=question)
        if filters:
            print(f"应用过滤条件:{filters}")
            relevant_chunks=self.retrieval_module.metadata_filtered_search(query=rewritten_query,filters=filters,top_k=self.config.top_k)
        else:
            relevant_chunks=self.retrieval_module.hybrid_search(rewritten_query,top_k=self.config.top_k)

        if relevant_chunks:
            chunk_info=[]
            for chunk in relevant_chunks:
                dish_name=chunk.metadata.get("dish_name","未知菜品")
                content_preview=chunk.page_content[:100].strip()
                if content_preview.startswith('#'):
                    title_end=content_preview.find('\n')if '\n' in content_preview else len(content_preview)
                    section_title=content_preview[:title_end].replace('#','').strip()
                    chunk_info.append(f"{dish_name}{section_title}")
                else:
                    chunk_info.append(f"{dish_name}(内容片段)")
            print(f"找到{len(relevant_chunks)}个相关文档块：{','.join(chunk_info)}")
        else:
            print(f"找到{len(relevant_chunks)}个相关文档块")
        if not relevant_chunks:
            return "抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。"
        
        #5.根据路由类型选择回答方式
        if router_type=="list":
            print("正在生成菜品列表")
            relevant_docs=self.data_module.get_parent_documents(relevant_chunks)
            doc_names=[]
            for doc in relevant_docs:
                dish_name=doc.metadata.get('dish_name','未知菜品')
                doc_names.append(dish_name)
            if doc_names:
                print(f"找到文档：{','.join(doc_names)}")

            return self.generation_module.generate_list_answer(query=question,context_docs=relevant_docs)
        else:
            print("正在生成菜品列表")
            relevant_docs=self.data_module.get_parent_documents(relevant_chunks)
            doc_names=[]
            for doc in relevant_docs:
                dish_name=doc.metadata.get('dish_name','未知菜品')
                doc_names.append(dish_name)
            if doc_names:
                print(f"找到文档：{','.join(doc_names)}")
            else:
                print(f"对应{len(relevant_docs)}个完整文档")
            print("生成详细回答")

            if router_type=="detail":
                if stream:
                    return self.generation_module.generate_step_by_step_answer_stream(query=question,context_docs=relevant_docs)
                else:
                    return self.generation_module.generate_step_by_step_answer(query=question,context_docs=relevant_docs)
                
            else:
                if stream:
                    return self.generation_module.generate_basic_answer_stream(query=question,context_docs=relevant_docs)
                else:
                    return self.generation_module.generate_basic_answer(query=question,context_docs=relevant_docs)
                
    def _extract_filters_from_query(self,query:str)->str:
        "从用户问题中提取元数据过滤条件"
        filters={}
        #分类关键词
        category_keywords=DataPreparationModule.get_supported_categories()
        for cat in category_keywords:
            if cat in query:
                filters['category']=cat
                break
        #难度关键词
        difficulty_words=DataPreparationModule.get_supported_difficulties()
        for diff in difficulty_words:
            if diff in query:
                filters['difficulty']=diff
                break

        return filters

    def search_by_category(self,category:str,query:str="")->List[str]:
        """
        按分类搜索菜品
        
        Args:
            category: 菜品分类
            query: 可选的额外查询条件
            
        Returns:
            菜品名称列表
        """
        if not self.retrieval_module:
            raise ValueError("请先构建知识库")
        #使用元数据过滤
        search_query=query if query else category
        filters={"category":category}
        docs=self.retrieval_module.metadata_filtered_search(search_query,filters,top_k=10)
        dish_names=[]
        for doc in docs:
            dish_name=doc.metadata.get('dish_name','未知菜品')
            if dish_name not in dish_names:
                dish_names.append(dish_name)
        return dish_names

    def get_ingredients_list(self,dish_name:str)->str:
        """
        获取指定菜品的食材信息

        Args:
            dish_name: 菜品名称

        Returns:
            食材信息
        """

        if not all(self.retrieval_module,self.generation_module):
            raise ValueError("请先构建知识库")
        #搜索相关文档
        docs=self.retrieval_module.hybrid_search(dish_name,top_k=3)

        #生成食材信息

        answer=self.generation_module.generate_basic_answer(f"{dish_name}需要什么食材?",docs)

        return answer
    
    def run_interactive(self):
        """运行交互式问答"""
        print("=" * 60)
        print("🍽️  尝尝咸淡RAG系统 - 交互式问答  🍽️")
        print("=" * 60)
        print("💡 解决您的选择困难症，告别'今天吃什么'的世纪难题！") 
        #初始化系统
        self.init_system()
        #构建知识库
        self.build_knowledge()
        print("\n交互式问答 (输入'退出'结束):")
        
        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if user_input.lower() in ['退出', 'quit', 'exit', '']:
                    break
                
                # 询问是否使用流式输出
                stream_choice = input("是否使用流式输出? (y/n, 默认y): ").strip().lower()
                use_stream = stream_choice != 'n'

                print("\n回答:")
                if use_stream:
                    # 流式输出
                    for chunk in self.ask_question(user_input, stream=True):
                        print(chunk, end="", flush=True)
                    print("\n")
                else:
                    # 普通输出
                    answer = self.ask_question(user_input, stream=False)
                    print(f"{answer}\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")
        
        print("\n感谢使用尝尝咸淡RAG系统！")
def main():
    """主函数"""
    try:
        # 创建RAG系统
        rag_system = RecipeRAGSystem()
        
        # 运行交互式问答
        rag_system.run_interactive()
        
    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        print(f"系统错误: {e}")

if __name__ == "__main__":
    main()
