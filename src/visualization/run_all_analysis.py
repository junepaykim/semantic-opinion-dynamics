import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

# 1. Base path and folder configuration
BASE_DIR = '/Users/hyunwooyu/Desktop/UCSD/ECE227/researcher B'
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

# Automatically create the 'figures' folder if it doesn't exist
os.makedirs(FIGURES_DIR, exist_ok=True)

# 2. Map information for the 4 networks to be analyzed
networks = [
    {"id": "Net1_ER", "agent": "Net1_agent_slices", "degroot": "Net1_ER_degroot_slices"},
    {"id": "Net2_SW", "agent": "Net2_agent_slices", "degroot": "Net2_SW_degroot_slices"},
    {"id": "Net3_SF", "agent": "Net3_agent_slices", "degroot": "Net3_SF_degroot_slices"},
    {"id": "Net4_KC", "agent": "Net4_KC_agent_slices", "degroot": "Net4_KC_degroot_slices"}
]

# 3. Load AI model and set Anchor points (Loaded only once for speed optimization)
print("Loading SBERT model... (This only takes time on the first run)")
model = SentenceTransformer('all-MiniLM-L6-v2')
anchor_remote = model.encode(["I fully support remote work and flexibility."])
anchor_rto = model.encode(["I believe everyone must return to the office full-time."])

def get_sentiment_score(emb):
    sim_remote = cosine_similarity(emb.reshape(1, -1), anchor_remote.reshape(1, -1))[0][0]
    sim_rto = cosine_similarity(emb.reshape(1, -1), anchor_rto.reshape(1, -1))[0][0]
    return sim_remote - sim_rto

# 4. Batch JSON data loading function
def load_json_data(folder_path, is_agent=True):
    all_data = []
    if not os.path.exists(folder_path):
        print(f"Cannot find the {folder_path} folder.")
        return pd.DataFrame()

    file_list = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    file_list = sorted(file_list, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))

    for file_name in file_list:
        with open(os.path.join(folder_path, file_name), 'r') as f:
            data = json.load(f)
            nodes = data.get('nodes', [])
            step = int(''.join(filter(str.isdigit, file_name)) or 0)
            for node in nodes:
                if is_agent:
                    all_data.append({'step': step, 'node_id': node['id'], 'opinion': node['prompt']})
                else:
                    all_data.append({'step': step, 'node_id': node['id'], 'opinionScore': node['opinionScore']})
    return pd.DataFrame(all_data)

# 5. Main automation loop (Iterate through each network to generate and save graphs)
for net in networks:
    print(f"\n========== [ Starting analysis for {net['id']} ] ==========")
    agent_path = os.path.join(BASE_DIR, net['agent'])
    degroot_path = os.path.join(BASE_DIR, net['degroot'])

    # --- [A] LLM Agent Analysis Part ---
    df_agent = load_json_data(agent_path, is_agent=True)
    if not df_agent.empty:
        print(f"[{net['id']}] Calculating text embeddings...")
        df_agent['embedding'] = list(model.encode(df_agent['opinion'].tolist()))
        df_agent['sentiment_score'] = df_agent['embedding'].apply(get_sentiment_score)

        # 1) Save Agent Line Plot
        plt.figure(figsize=(10, 6))
        for node_id in df_agent['node_id'].unique()[:10]:
            node_data = df_agent[df_agent['node_id'] == node_id]
            plt.plot(node_data['step'], node_data['sentiment_score'], label=f'Node {node_id}')
        plt.title(f'Opinion Drift: Remote vs. RTO ({net["id"]})')
        plt.xlabel('Time Step')
        plt.ylabel('Sentiment Score (Remote > 0 > RTO)')
        plt.axhline(0, color='black', linestyle='--', alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        line_save_path = os.path.join(FIGURES_DIR, f"{net['id']}_Agent_Line.jpg")
        plt.savefig(line_save_path)
        plt.close() # Clear memory (Important)
        print(f"Saved: {net['id']}_Agent_Line.jpg")

        # 2) Save Agent t-SNE Plot
        last_step_df = df_agent[df_agent['step'] == df_agent['step'].max()]
        if len(last_step_df) > 1:
            # Defensive code to automatically adjust perplexity based on the number of nodes
            p_value = min(30, max(1, len(last_step_df) - 1)) 
            embeddings_2d = TSNE(n_components=2, random_state=42, perplexity=p_value).fit_transform(np.stack(last_step_df['embedding'].values))
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=last_step_df['sentiment_score'], palette='coolwarm')
            plt.title(f'Semantic Clustering of Opinions ({net["id"]})')
            tsne_save_path = os.path.join(FIGURES_DIR, f"{net['id']}_Agent_tSNE.jpg")
            plt.savefig(tsne_save_path)
            plt.close()
            print(f"Saved: {net['id']}_Agent_tSNE.jpg")

    # --- [B] DeGroot Baseline Analysis Part ---
    df_degroot = load_json_data(degroot_path, is_agent=False)
    if not df_degroot.empty:
        plt.figure(figsize=(10, 6))
        for node_id in df_degroot['node_id'].unique()[:10]:
            node_data = df_degroot[df_degroot['node_id'] == node_id]
            plt.plot(node_data['step'], node_data['opinionScore'], label=f'Node {node_id}')
        plt.title(f'Opinion Convergence: DeGroot Baseline ({net["id"]})')
        plt.xlabel('Time Step')
        plt.ylabel('Opinion Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        degroot_save_path = os.path.join(FIGURES_DIR, f"{net['id']}_DeGroot_Line.jpg")
        plt.savefig(degroot_save_path)
        plt.close()
        print(f"Saved: {net['id']}_DeGroot_Line.jpg")

print("Analysis and saving for all networks are complete")