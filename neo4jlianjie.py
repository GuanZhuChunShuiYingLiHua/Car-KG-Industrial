import networkx as nx
from neo4j import GraphDatabase
import os

# --- 配置区 ---
GRAPHML_PATH = "./geely_local_kg/graph_chunk_entity_relation.graphml"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "ZD20040605"  # 👈 请修改为你的 Neo4j 密码

class GeelyGraphSyncer:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def sync(self, graphml_file):
        if not os.path.exists(graphml_file):
            print(f"❌ 错误：找不到文件 {graphml_file}")
            return

        # 1. 使用 networkx 读取 GraphML
        print(f"📖 正在读取图谱文件: {graphml_file}")
        G = nx.read_graphml(graphml_file)

        with self.driver.session() as session:
            # 2. 写入节点 (Nodes)
            print("⏳ 正在写入节点...")
            for node_id, attrs in G.nodes(data=True):
                # 提取属性，如果没有则设为未知
                entity_name = attrs.get('label', node_id)
                entity_type = attrs.get('entity_type', 'Entity')
                description = attrs.get('description', '')

                # 使用 MERGE 避免重复，并动态设置 Label
                cypher_node = f"""
                MERGE (n:`{entity_type}` {{name: $name}})
                SET n.description = $desc, n.last_updated = datetime()
                """
                session.run(cypher_node, name=entity_name, desc=description)

            # 3. 写入关系 (Edges)
            print("⏳ 正在写入关系...")
            for source, target, attrs in G.edges(data=True):
                rel_type = attrs.get('label', 'RELATED_TO').replace(" ", "_").upper()
                weight = attrs.get('weight', 1.0)
                description = attrs.get('description', '')

                # 构建关系，自动匹配源节点和目标节点
                cypher_rel = f"""
                MATCH (a {{name: $source}}), (b {{name: $target}})
                MERGE (a)-[r:`{rel_type}`]->(b)
                SET r.weight = $weight, r.description = $desc
                """
                session.run(cypher_rel, source=source, target=target, weight=weight, desc=description)

        print("✅ Neo4j 同步完成！你可以打开 Neo4j Browser 查看了。")

if __name__ == "__main__":
    syncer = GeelyGraphSyncer(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        syncer.sync(GRAPHML_PATH)
    finally:
        syncer.close()