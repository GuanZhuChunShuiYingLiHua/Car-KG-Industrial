import os
import asyncio
import nest_asyncio
import shutil
from functools import partial

# 核心：必须在 import lightrag 之前执行，防止 SSL 证书加载卡死
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['NO_PROXY'] = '127.0.0.1,localhost'
os.environ["PYTHONHTTPSVERIFY"] = "0" # 禁用 SSL 验证防止卡死

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

nest_asyncio.apply()

WORKING_DIR = "./geely_local_kg"
OLLAMA_HOST = "http://127.0.0.1:11434"

async def main():
    print(f"✅ 环境配置已重置。正在初始化吉利汽车图谱...")
    
    # 物理清理旧数据
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR)

    # 1. 解决 Relations 问题：使用 7B 模型
    # 2. 解决 500/NaN 问题：通过 partial 预设参数并限制并发
    fixed_llm_func = partial(ollama_model_complete, host=OLLAMA_HOST)
    fixed_embed_func = partial(ollama_embed, host=OLLAMA_HOST, embed_model="bge-m3:latest")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=fixed_llm_func,
        llm_model_name="qwen2.5:7b", # 升级为 7b
        llm_model_max_async=1,       # 极其重要：本地单并发最稳
        
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=fixed_embed_func
        ),
        embedding_func_max_async=1,  # 解决 500 NaN 报错的关键
    )

    await rag.initialize_storages()
    
    # 这里的文本尽量写得清晰，方便 7B 模型提取关系
    test_content = """
    吉利汽车研究院负责极氪001的整车架构设计。
    极氪001的核心组件包括由吉利自研的高性能逆变器。
    动力电池由供应商宁德时代(CATL)提供，而热管理系统则由博世(Bosch)供应。
    """

    print("🚀 正在提取实体与关系（使用 7B 模型，可能需要 1-2 分钟，请稍候）...")
    await rag.ainsert(test_content)
    
    # 验证是否成功
    print("\n🧐 正在执行测试查询...")
    # 3. 解决 Rerank 警告：显式关闭 enable_rerank
    query = "总结极氪001的供应链结构。"
    response = await rag.aquery(query, param=QueryParam(
        mode="naive", 
        enable_rerank=False 
    ))
    
    print("\n" + "="*50)
    print(f"图谱回答:\n{response}")
    print("="*50)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n用户手动停止。")
    except Exception as e:
        print(f"❌ 运行报错: {e}")