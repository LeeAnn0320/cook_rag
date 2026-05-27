"""
检索和优化模块
"""
import logging

logger=logging.getLogger(__name__)

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from typing import List,Dict,Any
class RetrievalOptimizationModule:
    """检索优化模块  负责混合检索和过滤"""
    def __init__(self,vectorstore:FAISS,chunks:List[Document]):
        
        """
        初始化检索优化模块

        Args:
            vectorstore:FAISS向量存储
            chunks:文档块列表
    
        """
        if vectorstore is None:
            raise ValueError("vectorstore 不能为 None，请先构建索引")
        self.vectorstore=vectorstore
        self.chunks=chunks
        self.setup_retrievers()
    
    def setup_retrievers(self):
        """设置向量检索器和BM25检索器"""
        logger.info("正在设置检索器")
        #向量检索器
        self.vector_retriever=self.vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":5})
        #BM25检索器
        self.bm25_retriever=BM25Retriever.from_documents(
            self.chunks,k=5
        )


    def hybrid_search(self,query:str,top_k:int=3)->List[Document]:
        """
        混合检索 - 结合向量检索和BM25检索，使用RRF重排

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            检索
        """
        #分别获取向量检索和BM26检索结果
        vector_docs=self.vector_retriever.invoke(query)
        bm25_docs=self.bm25_retriever.invoke(query)

        #使用后RRF重排
        reranked_docs=self._rrf_rerank(vector_docs,bm25_docs)
        return reranked_docs[:top_k]

    def metadata_filtered_search(self,query:str,filters:Dict[str,Any],top_k:int=5)->List[Document]:
        """
        带元数据过滤的检索
        
        Args:
            query: 查询文本
            filters: 元数据过滤条件
            top_k: 返回结果数量
            
        Returns:
            过滤后的文档列表
        """

        #先进行混合检索 获得更多候选

        docs=self.hybrid_search(query=query,top_k=top_k)
        #应用元数据过滤
        filtered_docs=[]
        for doc in docs:
            match=True
            for key,value in filters.items():
                if key in doc.metadata:
                    if isinstance(value,list):
                        if doc.metadata[key] not in value:
                            match=False
                    else:
                        if doc.metadata[key]!=value:
                            match=False
                            break
                else:
                    match=False
                    break
            if match:
                filtered_docs.append(doc)
                if len(filtered_docs)>=top_k:
                    break
        return filtered_docs
    def _rrf_rerank(self,vector_docs:List[Document],bm25_docs:List[Document],k:int=60)->List[Document]:
        """
        使用RRF (Reciprocal Rank Fusion) 算法重排文档

        Args:
            vector_docs: 向量检索结果
            bm25_docs: BM25检索结果
            k: RRF参数，用于平滑排名

        Returns:
            重排后的文档列表
        """
        doc_scores={}
        doc_object={}

        #计算向量检索结果的RRF分数
        for rank,doc in enumerate(vector_docs):
            #使用文档内容的哈希作为唯一标识
            doc_id=hash(doc.page_content)
            doc_object[doc_id]=doc
            # RRF公式: 1 / (k + rank)
            rrf_score=1.0/(k+rank+1)
            doc_scores[doc_id]=doc_scores.get(doc_id,0)+rrf_score

            logger.debug(f"向量检索-文档{rank+1}:RRF分数={rrf_score:.4f}")

        #计算BM25检索结果的分数
        for rank,doc in enumerate(bm25_docs):
            doc_id=hash(doc.page_content)
            doc_object[doc_id]=doc
            # RRF公式: 1 / (k + rank)
            rrf_score=1.0/(k+rank+1)
            doc_scores[doc_id]=doc_scores.get(doc_id,0)+rrf_score

            logger.debug(f"文档检索-文档{rank+1}:RRF分数={rrf_score:.4f}")

        #按最终RFF分数排序
        sorted_docs=sorted(doc_scores.items(),key=lambda x:x[1],reverse=True)
        #构建最终结果
        reranked_docs=[]
        for doc_id,final_score in sorted_docs:
            if doc_id in doc_object:
                doc=doc_object[doc_id]
                #将RRF分数添加到文档元数据中
                doc.metadata['rrf_score']=final_score
                reranked_docs.append(doc)
                logger.debug(f"RRF完成：向量检索{len(vector_docs)}个文档，BM25检索{len(bm25_docs)}个文档,合并后{len(reranked_docs)}个文档")
        return reranked_docs               