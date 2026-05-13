"""
临床决策辅助系统 - 图索引模块
实现医疗实体和关系的键值对结构 (K,V)
"""

import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

@dataclass
class EntityKeyValue:
    """实体键值对"""
    entity_name: str
    index_keys: List[str]  # 用来被搜索的关键词列表
    value_content: str     # 拼接好的结构化文本
    entity_type: str       # 实体类型 (Department, Consultation, Disease)
    metadata: Dict[str, Any]

@dataclass
class RelationKeyValue:
    """关系键值对"""
    relation_id: str
    index_keys: List[str]  # 用来被搜索的关键词列表
    value_content: str     # 关系描述内容
    relation_type: str     # 关系类型
    source_entity: str     # 源实体
    target_entity: str     # 目标实体
    metadata: Dict[str, Any]

class MedicalGraphIndexingModule:
    """
    医学图索引模块
    核心功能：
    1. 为科室、问答记录、疾病创建键值对
    2. 为医学关系（如：属于科室、涉及疾病）创建键值对
    """

    def __init__(self, config, llm_client):
        self.config = config
        self.llm_client = llm_client

        self.entity_kv_store: Dict[str, EntityKeyValue] = {}
        self.relation_kv_store: Dict[str, RelationKeyValue] = {}

        self.key_to_entities: Dict[str, List[str]] = defaultdict(list)
        self.key_to_relations: Dict[str, List[str]] = defaultdict(list)

    def create_entity_key_values(self, departments: List[Any], consultations: List[Any], diseases: List[Any] = None) -> Dict[str, EntityKeyValue]:
        """
        为医疗实体创建键值对结构
        """
        logger.info("开始创建医疗实体键值对...")

        # 1. 处理科室实体 (Department)
        for dept in departments:
            entity_id = getattr(dept, 'node_id', dept.get('node_id', f"dept_{dept['name']}"))
            entity_name = getattr(dept, 'name', dept.get('name', '未知科室'))

            content_parts = [f"临床科室: {entity_name}"]

            entity_kv = EntityKeyValue(
                entity_name=entity_name,
                index_keys=[entity_name, f"{entity_name}门诊", f"{entity_name}医生"],
                value_content='\n'.join(content_parts),
                entity_type="Department",
                metadata={"node_id": entity_id}
            )

            self.entity_kv_store[entity_id] = entity_kv
            for key in entity_kv.index_keys:
                self.key_to_entities[key].append(entity_id)

        # 2. 处理问答记录实体 (Consultation - 来自CSV的一条数据)
        for consult in consultations:
            # 兼容对象属性或字典键值
            entity_id = getattr(consult, 'node_id', consult.get('node_id'))
            title = getattr(consult, 'title', consult.get('title', '未知咨询'))
            ask = getattr(consult, 'ask', consult.get('ask', ''))
            answer = getattr(consult, 'answer', consult.get('answer', ''))
            dept_name = getattr(consult, 'department', consult.get('department', ''))

            entity_name = f"病例_{entity_id}"

            content_parts = [
                f"【主诉/标题】: {title}",
                f"【所属科室】: {dept_name}",
                f"【患者描述】: {ask}",
                f"【医生诊断/建议】: {answer}"
            ]

            # 提取标题中的潜在关键词作为索引键
            index_keys = [entity_name, title]

            entity_kv = EntityKeyValue(
                entity_name=entity_name,
                index_keys=index_keys,
                value_content='\n'.join(content_parts),
                entity_type="Consultation",
                metadata={"node_id": entity_id, "department": dept_name}
            )

            self.entity_kv_store[entity_id] = entity_kv
            for key in entity_kv.index_keys:
                self.key_to_entities[key].append(entity_id)

        # 3. 处理疾病实体 (Disease) —— 这里是新增的内容
        if diseases:
            for disease in diseases:
                entity_id = disease.get('node_id')
                entity_name = disease.get('name', '未知疾病')
                content_parts = [f"临床疾病/症状: {entity_name}"]
                entity_kv = EntityKeyValue(
                    entity_name=entity_name,
                    index_keys=[entity_name, f"{entity_name}症状", f"{entity_name}治疗"],
                    value_content='\n'.join(content_parts),
                    entity_type="Disease",
                    metadata={"node_id": entity_id}
                )
                self.entity_kv_store[entity_id] = entity_kv
                for key in entity_kv.index_keys:
                    self.key_to_entities[key].append(entity_id)

        logger.info(f"医疗实体键值对创建完成，共 {len(self.entity_kv_store)} 个实体")
        return self.entity_kv_store

    def create_relation_key_values(self, relationships: List[Tuple[str, str, str]]) -> Dict[str, RelationKeyValue]:
        """
        为医学关系创建键值对结构
        """
        logger.info("开始创建医学关系键值对...")

        for i, (source_id, relation_type, target_id) in enumerate(relationships):
            relation_id = f"rel_{i}_{source_id}_{target_id}"

            source_entity = self.entity_kv_store.get(source_id)
            target_entity = self.entity_kv_store.get(target_id)

            if not source_entity or not target_entity:
                continue

            content_parts = [
                f"临床逻辑: {relation_type}",
                f"源对象: {source_entity.entity_name} ({source_entity.entity_type})",
                f"目标对象: {target_entity.entity_name} ({target_entity.entity_type})"
            ]

            # 生成医疗关系索引键
            index_keys = self._generate_medical_relation_keys(
                source_entity, target_entity, relation_type
            )

            relation_kv = RelationKeyValue(
                relation_id=relation_id,
                index_keys=index_keys,
                value_content='\n'.join(content_parts),
                relation_type=relation_type,
                source_entity=source_id,
                target_entity=target_id,
                metadata={
                    "source_name": source_entity.entity_name,
                    "target_name": target_entity.entity_name,
                    "created_from_graph": True
                }
            )

            self.relation_kv_store[relation_id] = relation_kv
            for key in index_keys:
                self.key_to_relations[key].append(relation_id)

        return self.relation_kv_store

    def _generate_medical_relation_keys(self, source: EntityKeyValue, target: EntityKeyValue, rel_type: str) -> List[str]:
        """生成医学主题相关的索引键"""
        keys = [rel_type]

        if rel_type == "BELONGS_TO_DEPT":
            keys.extend(["科室归属", "就诊指南", target.entity_name])
        elif rel_type == "MENTIONS_DISEASE" or rel_type == "HAS_SYMPTOM":
            keys.extend(["临床症状", "疾病诊断", target.entity_name, "表现为"])

        return list(set(keys))

    def _llm_enhance_relation_keys(self, source_entity: EntityKeyValue,
                                   target_entity: EntityKeyValue,
                                   relation_type: str) -> List[str]:
        """
        使用LLM增强临床关系索引键，生成医学全局主题和同义词
        """
        prompt = f"""
        作为医学专家，分析以下临床实体关系，生成相关的主题关键词或医学同义词：

        源实体: {source_entity.entity_name} ({source_entity.entity_type})
        目标实体: {target_entity.entity_name} ({target_entity.entity_type})
        临床关系: {relation_type}

        请生成3-5个相关的医学主题关键词（如症状别名、相关并发症、科室通俗叫法等），用于医疗问答的索引和检索。
        返回JSON格式：{{"keywords": ["关键词1", "关键词2", "关键词3"]}}
        """

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
                response_format={"type": "json_object"}  # 建议加上强制JSON输出
            )

            # 使用更健壮的JSON解析
            raw_content = response.choices[0].message.content.strip()
            import re
            json_match = re.search(r'\{[\s\S]*\}', raw_content)
            clean_content = json_match.group(0) if json_match else raw_content

            result = json.loads(clean_content)
            return result.get("keywords", [])

        except Exception as e:
            logger.error(f"LLM增强临床关系索引键失败: {e}")
            return []
    
    def deduplicate_entities_and_relations(self):
        """
        去重相同的实体和关系，优化图操作
        """
        logger.info("开始去重实体和关系...")
        
        # 实体去重：基于名称
        name_to_entities = defaultdict(list)
        for entity_id, entity_kv in self.entity_kv_store.items():
            name_to_entities[entity_kv.entity_name].append(entity_id)
        
        # 合并重复实体
        entities_to_remove = []
        for name, entity_ids in name_to_entities.items():
            if len(entity_ids) > 1:
                # 保留第一个，合并其他的内容
                primary_id = entity_ids[0]
                primary_entity = self.entity_kv_store[primary_id]
                
                for entity_id in entity_ids[1:]:
                    duplicate_entity = self.entity_kv_store[entity_id]
                    # 合并内容
                    primary_entity.value_content += f"\n\n补充信息: {duplicate_entity.value_content}"
                    # 标记删除
                    entities_to_remove.append(entity_id)
        
        # 删除重复实体
        for entity_id in entities_to_remove:
            del self.entity_kv_store[entity_id]
        
        # 关系去重：基于源-目标-类型
        relation_signature_to_ids = defaultdict(list)
        for relation_id, relation_kv in self.relation_kv_store.items():
            signature = f"{relation_kv.source_entity}_{relation_kv.target_entity}_{relation_kv.relation_type}"
            relation_signature_to_ids[signature].append(relation_id)
        
        # 合并重复关系
        relations_to_remove = []
        for signature, relation_ids in relation_signature_to_ids.items():
            if len(relation_ids) > 1:
                # 保留第一个，删除其他
                for relation_id in relation_ids[1:]:
                    relations_to_remove.append(relation_id)
        
        # 删除重复关系
        for relation_id in relations_to_remove:
            del self.relation_kv_store[relation_id]
        
        # 重建索引映射
        self._rebuild_key_mappings()
        
        logger.info(f"去重完成 - 删除了 {len(entities_to_remove)} 个重复实体，{len(relations_to_remove)} 个重复关系")
    
    def _rebuild_key_mappings(self):
        """重建键到实体/关系的映射"""
        self.key_to_entities.clear()
        self.key_to_relations.clear()
        
        # 重建实体映射
        for entity_id, entity_kv in self.entity_kv_store.items():
            for key in entity_kv.index_keys:
                self.key_to_entities[key].append(entity_id)
        
        # 重建关系映射
        for relation_id, relation_kv in self.relation_kv_store.items():
            for key in relation_kv.index_keys:
                self.key_to_relations[key].append(relation_id)
    
    def get_entities_by_key(self, key: str) -> List[EntityKeyValue]:
        """根据索引键获取实体"""
        entity_ids = self.key_to_entities.get(key, [])
        return [self.entity_kv_store[eid] for eid in entity_ids if eid in self.entity_kv_store]
    
    def get_relations_by_key(self, key: str) -> List[RelationKeyValue]:
        """根据索引键获取关系"""
        relation_ids = self.key_to_relations.get(key, [])
        return [self.relation_kv_store[rid] for rid in relation_ids if rid in self.relation_kv_store]

    def get_statistics(self) -> Dict[str, Any]:
        """获取临床键值对存储统计信息"""
        return {
            "total_entities": len(self.entity_kv_store),
            "total_relations": len(self.relation_kv_store),
            "total_entity_keys": sum(len(kv.index_keys) for kv in self.entity_kv_store.values()),
            "total_relation_keys": sum(len(kv.index_keys) for kv in self.relation_kv_store.values()),
            "entity_types": {
                # 替换为医疗系统的三大核心实体
                "Department": len([kv for kv in self.entity_kv_store.values() if kv.entity_type == "Department"]),
                "Consultation": len([kv for kv in self.entity_kv_store.values() if kv.entity_type == "Consultation"]),
                "Disease": len([kv for kv in self.entity_kv_store.values() if kv.entity_type == "Disease"])
            }
        }