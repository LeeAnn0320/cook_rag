"""
索引构建模块
"""

import logging
logger=logging.getLogger(__name__)
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
class IndexConstructionModule:
    
    def __init__(self,model_name:str="BAAI/bge-small-zh-v1.5",index_save_path:str="./vector_index"):

        self.model_name=model_name
        self.index_save_path=index_save_path
        self.embeddings=None
        self.vectorstore=None

        self.setup_embeddings()
    def setup_embeddings(self):
        """初始化嵌入模型"""
        logger.info(f"正在初始化嵌入模型{self.model_name}")


        self.embeddings=HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device':'cpu'},
            encode_kwargs={'normalize_embedding':True}
        )

        logger.info("嵌入模型初始化完成")

    def build_vector_index(self,chunks:List[Document])->FAISS:
        """构建向量索引
        
        Args:
            chunks:文档块列表
        Return:
            FAISS向量存储对象

        """
        self.vectorstore=FAISS.from_documents(
            documents=chunks,embedding=self.embeddings
        )
        logger.info(f"向量构建索引完成，包含{len(chunks)}个向量")
        return self.vectorstore
    
    def add_documents(self,new_chunks:List[Document]):
        """
        向现有索引添加新文档
        """

        if not self.vectorstore:
            raise ValueError("请先构建向量索引")

        logger.info(f"正在添加{len(new_chunks)}个新文档到索引")
        self.vectorstore.add_documents(new_chunks)
        logger.info("新文档添加完成")

    def save_index(self):
        """保存向量索引到配置文件的路径"""
        if not self.vectorstore:
            raise ValueError("请先构建向量索引")
        
        #确保目录存在
        Path(self.index_save_path).mkdir(parents=True,exist_ok=True)
        self.vectorstore.save_local(self.index_save_path)
        logger.info(f"向量索引已经添加到{self.index_save_path}")
    def load_index(self):
        """从配置文件加载向量索引"""
        if not self.embeddings:
            self.setup_embeddings()

        if not Path(self.index_save_path).exists():
            logger.info(f"索引路径不存在:{self.index_save_path}")
            return None
        try:
            self.vectorstore.load_local(
                self.index_save_path,self.embeddings,allow_dangerous_deserialization=True
            )
            return self.vectorstore
        except Exception as e:
            logger.warning(f"向量索引加载失败:{e}")
            return None
        

    def similarity_search(self,query:str,k:int=5)->List[Document]:
        """相似度搜索
        Args:
            query:查询文本
            k:返回结果数量        
        Return:
            相似文档列表
        """

        if not self.vectorstore:
            raise ValueError("请先构建或加载向量索引")
        return self.vectorstore.similarity_search(query=query,k=k)
        
