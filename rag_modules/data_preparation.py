"""
数据准备模块
"""

import logging
import hashlib
logger=logging.getLogger(__name__)

from pathlib import Path
from typing import List,Dict,Optional,Any
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
import uuid
class DataPreparationModule:
    """数据准备模块，负责 数据加载 清洗 预处理"""
    CATEGORY_MAPPING = {
        'meat_dish': '荤菜',
        'vegetable_dish': '素菜',
        'soup': '汤品',
        'dessert': '甜品',
        'breakfast': '早餐',
        'staple': '主食',
        'aquatic': '水产',
        'condiment': '调料',
        'drink': '饮品'
    }
    CATEGORY_LABELS = list(set(CATEGORY_MAPPING.values()))
    DIFFICULTY_LABELS = ['非常简单', '简单', '中等', '困难', '非常困难']

    def __init__(self,data_path:str):
        self.data_path=data_path
        self.documents:List[Document]=[]#父文档
        self.chunks:List[Document]=[]#子文档
        self.parent_child_map:Dict[str,str]={}#子文档id->父文档id的映射


    def load_document(self)->List[Document]:
        """加载文档"""

        logger.info(f"正在从{self.data_path}加载文档")
        documents=[]
        data_path_obj=Path(self.data_path)
        for md_file in data_path_obj.rglob("*.md"):
            try:
                with open(md_file,'r',encoding='utf-8') as f:
                    content=f.read()
                try:
                    #获得文档的绝对路径
                    data_root=Path(self.data_path).resolve()
                    #获得相对路径
                    relative_path=Path(md_file).resolve().relative_to(data_root).as_posix()
                except Exception:
                    relative_path=Path(md_file).as_posix()
                    #使用 MD5 哈希将路径转为 32 位十六进制字符串
                parent_id=hashlib.md5(relative_path.encode("utf-8")).hexdigest()
                    
                #创建Document对象
                doc=Document(
                    page_content=content,
                    metadata={
                        "source":str(md_file),
                        "parent_id":parent_id,
                        "doc_type":"parent"#标记为父文档
                    }
                )
                documents.append(doc)
            except Exception as e:
                logging.warning(f"读取文件{md_file}失败：{e}")

        for doc in documents:
            self._enchance_metadata(doc=doc)
        self.documents=documents
        logger.info(f"成功加载{len(documents)}个文档")
        return documents


    def _enchance_metadata(self,doc:Document):
        """增强文档元数据"""
        file_path=Path(doc.metadata.get("source",""))
        path_parts=file_path.parts
        #提取菜品分类
        doc.metadata["category"]="其他"
        for key,value in self.CATEGORY_MAPPING.items():
            if key in path_parts:
                doc.metadata["category"]=value
                break
        # 提取菜品名称
        doc.metadata['dish_name'] = file_path.stem

        # 分析难度等级
        content = doc.page_content
        if '★★★★★' in content:
            doc.metadata['difficulty'] = '非常困难'
        elif '★★★★' in content:
            doc.metadata['difficulty'] = '困难'
        elif '★★★' in content:
            doc.metadata['difficulty'] = '中等'
        elif '★★' in content:
            doc.metadata['difficulty'] = '简单'
        elif '★' in content:
            doc.metadata['difficulty'] = '非常简单'
        else:
            doc.metadata['difficulty'] = '未知'

    @classmethod #无需创建实例即可调用
    def get_supported_categories(cls)->List[str]:
        """对外提供支持的分类标签列表"""
        return cls.CATEGORY_LABELS
    @classmethod
    def get_supported_difficulties(cls)->List[str]:
        """对外提供支持的难度标签列表"""
        return cls.DIFFICULTY_LABELS
    
    def chunk_documents(self)->List[Document]:
        """
        文档切块
        
        Return:
            切块后的文档列表
        """
        logger.info(f"正在进行文档感知分块")
        if not self.documents:
            raise("请先加载文档")
        #使用markdown标题分割器
        chunks=self._markdown_header_split()
        for i,chunk in enumerate(chunks):
            if 'chunk_id' not in chunk.metadata:
                chunk.metadata['chunk_id']=str(uuid.uuid4())
            chunk.metadata['batch_index']=i
            chunk.metadata['chunk_size']=len(chunk.page_content)
        self.chunks=chunks
        logger.info(f"Markdown分割完成，共生成{len(self.chunks)}个chunk")
        return chunks
    def _markdown_header_split(self)->List[Document]:
        """
        使用Markdown标题分割器进行结构化分割

        Returns:
            按标题结构分割的文档列表
        """

        headers_to_split_on=[
            ('#',"一级标题"),
            ('##',"二级标题"),
            ('###',"三级标题")
        ]

        #创建markdown分割器
        markdown_spliter=MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )
        all_chunks=[]
        for doc in self.documents:
            try:
                content_preview=doc.page_content[:200]
                has_headers=any(line.strip().startswith('#')for line in content_preview.split('\n'))

                if not has_headers:
                    logger.warning(f"文档:{doc.metadata.get('dish_name','未知')}内容中没有发现标题")
                    logger.debug(f"内容预览：{content_preview}")

                md_chunks=markdown_spliter.split_text(doc.page_content)
                logger.debug(f"文档:{doc.metadata.get('dish_name','未知')}被切成了{len(md_chunks)}块")
                if len(md_chunks)<=1:
                    logger.debug(f"文档:{doc.metadata.get('dish_name','未知')}未能按标题分割")
                #为每个子文档建立与父文档的关系

                parent_id=doc.metadata["parent_id"]
                for i,chunk in enumerate(md_chunks):
                    #为每个子块分配唯一ID
                    child_id=str(uuid.uuid4())
                    chunk.metadata.update(doc.metadata)
                    chunk.metadata.update({
                        "chunk_id":child_id,
                        "doc_type":"child",
                        "parent_id":parent_id,
                        "chunk_index":i
                    })
                    self.parent_child_map[child_id]=parent_id
                all_chunks.extend(md_chunks)
            except Exception as e:
                logger.warning(f"文档 {doc.metadata.get('source', '未知')} Markdown分割失败: {e}")
                # 如果Markdown分割失败，将整个文档作为一个chunk
                all_chunks.append(doc)
        logger.info(f"Markdown结构分割完成，生成 {len(all_chunks)} 个结构化块")
        return all_chunks
    
    def filter_documents_by_category(self,category:str)->List[Document]:

        "根据菜品分类过滤文档"
        return[doc for doc in self.documents if doc.metadata.get("category")==category]
    
    def filter_documents_by_difficuty(self,difficulty:str)->List[Document]:
        "根据难度过滤文档"
        return[doc for doc in self.documents if doc.metadata.get("difficulty")==difficulty]
    
    def get_statics(self)->Dict[str,Any]:
        """
        获取数据统计信息
        """

        if not self.documents:
            return {}
        categories={}
        difficulties={}

        for doc in self.documents:
            #统计分类
            category=doc.metadata.get('category','未知')
            categories[category]=categories.get('category',0)+1

            #统计难度
            difficulty=doc.metadata.get('difficulty','未知')
            difficulties[difficulty]=difficulties.get('difficulty',0)+1
        return {
            'total_documents':len(self.documents),
            'total_chunks':len(self.chunks),
            'categories':categories,
            'difficulties':difficulties,
            "avg_chunk_size":sum(chunk.metadata.get('chunk_size',0)for chunk in self.chunks) / len(self.chunks) if self.chunks else 0

        }
    
    def export_metadata(self, output_path: str):
        """
        导出元数据到JSON文件
        
        Args:
            output_path: 输出文件路径
        """
        import json
        
        metadata_list = []
        for doc in self.documents:
            metadata_list.append({
                'source': doc.metadata.get('source'),
                'dish_name': doc.metadata.get('dish_name'),
                'category': doc.metadata.get('category'),
                'difficulty': doc.metadata.get('difficulty'),
                'content_length': len(doc.page_content)
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)
        
        logger.info(f"元数据已导出到: {output_path}")

    def get_parent_documents(self,child_chunks:List[Document])->List[Document]:
        """
        根据子块获取对应父文档
        
        Args:
            child_chunks：检索到的子块列表
        Return:
            对应的父文档列表 （去重 按相关性排序）
        """

        parent_relevance={}
        parent_docs_map={}
        #收集所有相关的父文档ID和相关性分数
        for chunk in child_chunks:
            parent_id=chunk.metadata.get("parent_id")
            if parent_id:
                parent_relevance[parent_id]=parent_relevance.get(parent_id,0)+1
                #缓存父文档 避免重复查找
                if parent_id not in parent_docs_map:
                    for doc in self.documents:
                        if doc.metadata.get("parent_id")==parent_id:
                            parent_docs_map[parent_id]=doc
                            break
        #按相关性排序  匹配次数多的排前面

        sorted_parent_ids=sorted(parent_relevance,key=lambda x:parent_relevance[x],reverse=True)

        #构建去重后的父文档列表
        parent_docs=[]
        for parent_id in sorted_parent_ids:
            if parent_id in parent_docs_map:
                parent_docs.append(parent_docs_map[parent_id])

        # 收集父文档名称和相关性信息用于日志
        parent_info = []
        for doc in parent_docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            parent_id = doc.metadata.get('parent_id')
            relevance_count = parent_relevance.get(parent_id, 0)
            parent_info.append(f"{dish_name}({relevance_count}块)")

        logger.info(f"从 {len(child_chunks)} 个子块中找到 {len(parent_docs)} 个去重父文档: {', '.join(parent_info)}")
        return parent_docs