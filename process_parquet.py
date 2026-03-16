import pandas as pd
import pyarrow.parquet as pq
import json
import numpy as np
from typing import Dict, List, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial, lru_cache
import os
from collections import Counter
import time
from pathlib import Path
import psutil
import threading
import queue
import math

# =========================
# 1. 基础映射与工具函数
# =========================

TILE_MAPPING = {
    i: tile for i, tile in enumerate([
        "1万", "2万", "3万", "4万", "5万", "6万", "7万", "8万", "9万",
        "1筒", "2筒", "3筒", "4筒", "5筒", "6筒", "7筒", "8筒", "9筒",
        "1索", "2索", "3索", "4索", "5索", "6索", "7索", "8索", "9索",
        "东", "南", "西", "北", "白", "发", "中"
    ])
}

TILE_NAME_TO_INDEX = {name: index for index, name in TILE_MAPPING.items()}
ROUND_WIND_MAPPING = {0: "东", 1: "南", 2: "西", 3: "北"}

def read_parquet_file(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_parquet(file_path)
        print(f"成功读取文件: {file_path}")
        print(f"数据形状: {df.shape}")
        return df
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

def convert_tile_index_to_name(tile_index: int) -> str:
    if 0 <= tile_index <= 33:
        return TILE_MAPPING.get(tile_index, f"未知牌({tile_index})")
    return f"无效索引({tile_index})"

def convert_tile_name_to_index(tile_name: str) -> int:
    return TILE_NAME_TO_INDEX.get(tile_name, -1)

def convert_round_wind_to_name(round_wind: int) -> str:
    return ROUND_WIND_MAPPING.get(round_wind, f"未知场风({round_wind})")

def _counts_dict_to_list(tile_counts: Dict[int,int]) -> List[int]:
    arr = [0]*34
    for k,v in tile_counts.items():
        if 0 <= k < 34:
            arr[k] = v
    return arr

def calculate_dora_tiles(dora_indicators: List[int]) -> List[int]:
    """
    将宝牌指示牌转换为实际的宝牌。
    规则：
    - 数牌：n -> n+1 (9 -> 1)
    - 风牌：东(27)->南(28)->西(29)->北(30)->东(27)
    - 三元：白(31)->发(32)->中(33)->白(31)
    """
    dora_tiles = []
    for indicator in dora_indicators:
        if indicator < 0 or indicator > 33: continue
        
        dora_tile = -1
        
        # 万子 (0-8)
        if 0 <= indicator <= 8:
            dora_tile = (indicator + 1) % 9
        # 筒子 (9-17)
        elif 9 <= indicator <= 17:
            dora_tile = 9 + ((indicator - 9 + 1) % 9)
        # 索子 (18-26)
        elif 18 <= indicator <= 26:
            dora_tile = 18 + ((indicator - 18 + 1) % 9)
        # 风牌 (27-30): 东南西北
        elif 27 <= indicator <= 30:
            dora_tile = 27 + ((indicator - 27 + 1) % 4)
        # 三元牌 (31-33): 白发中
        elif 31 <= indicator <= 33:
            dora_tile = 31 + ((indicator - 31 + 1) % 3)
            
        if dora_tile != -1:
            dora_tiles.append(dora_tile)
            
    return list(set(dora_tiles)) # 去重

def convert_counts_to_names(counts: List[int]) -> Dict[str, int]:
    result = {}
    for i, count in enumerate(counts):
        if count > 0:
            tile_name = convert_tile_index_to_name(i)
            result[tile_name] = count
    return result

# =========================
# 2. 牌效逻辑 (Shanten / Ukeire)
# =========================

def calculate_shanten(tile_counts: Dict[int,int], meld_count: int = 0) -> int:
    cnt = _counts_dict_to_list(tile_counts)
    manzu = cnt[0:9]; pinzu = cnt[9:18]; sozu = cnt[18:27]; jihai = cnt[27:34]
    hai_arr = [manzu, pinzu, sozu, jihai]
    s_normal = _syanten_standard(hai_arr, meld_count)
    s_7 = _syanten7(hai_arr)
    s_13 = _syanten13(hai_arr)
    return min(s_normal, s_7, s_13)

def _syanten_standard(hai_arr, meld_count: int = 0):
    @lru_cache(maxsize=8192)
    def suit_calc(a_tuple):
        a = list(a_tuple)
        best = (0,0)
        idx = 0
        while idx < len(a) and a[idx] == 0: idx += 1
        if idx >= len(a): return best
        if a[idx] >= 3:
            a[idx] -= 3
            r = suit_calc(tuple(a))
            candidate = (r[0]+1, r[1])
            if candidate[0]*10 + candidate[1] > best[0]*10 + best[1]: best = candidate
            a[idx] += 3
        if len(a) == 9:
            if idx <= 6 and a[idx] > 0 and a[idx+1] > 0 and a[idx+2] > 0:
                a[idx] -= 1; a[idx+1] -= 1; a[idx+2] -= 1
                r = suit_calc(tuple(a))
                candidate = (r[0]+1, r[1])
                if candidate[0]*10 + candidate[1] > best[0]*10 + best[1]: best = candidate
                a[idx] += 1; a[idx+1] += 1; a[idx+2] += 1
            if idx <= 7 and a[idx] > 0 and a[idx+1] > 0:
                a[idx] -= 1; a[idx+1] -= 1
                r = suit_calc(tuple(a))
                candidate = (r[0], r[1]+1)
                if candidate[0]*10 + candidate[1] > best[0]*10 + best[1]: best = candidate
                a[idx] += 1; a[idx+1] += 1
            if idx <= 6 and a[idx] > 0 and a[idx+2] > 0:
                a[idx] -= 1; a[idx+2] -= 1
                r = suit_calc(tuple(a))
                candidate = (r[0], r[1]+1)
                if candidate[0]*10 + candidate[1] > best[0]*10 + best[1]: best = candidate
                a[idx] += 1; a[idx+2] += 1
        if a[idx] >= 2:
            a[idx] -= 2
            r = suit_calc(tuple(a))
            candidate = (r[0], r[1]+1)
            if candidate[0]*10 + candidate[1] > best[0]*10 + best[1]: best = candidate
            a[idx] += 2
        return best

    m_total = 0; t_total = 0
    for s_idx, suit in enumerate(hai_arr):
        if s_idx < 3: m,t = suit_calc(tuple(suit))
        else: m = sum(x//3 for x in suit); t = sum(1 for x in suit if x%3==2)
        m_total += m; t_total += t
    shanten = 8 - 2*m_total - t_total
    if shanten < 0: shanten = 0
    return shanten

def _syanten7(hai_arr):
    arr = hai_arr[0] + hai_arr[1] + hai_arr[2] + hai_arr[3]
    pairs = sum(1 for x in arr if x >= 2)
    singles = sum(1 for x in arr if x == 1)
    if pairs + singles >= 7: return 6 - pairs
    else: return 6 - pairs + (7 - pairs - singles)

def _syanten13(hai_arr):
    arr = [hai_arr[0][0], hai_arr[0][8], hai_arr[1][0], hai_arr[1][8],
           hai_arr[2][0], hai_arr[2][8]] + hai_arr[3]
    unique = sum(1 for x in arr if x > 0)
    has_pair = 1 if any(x > 1 for x in arr) else 0
    return 13 - unique - has_pair

@lru_cache(maxsize=4096)
def _calculate_ukeire_cached(tile_counts_tuple: tuple, base_shanten: int, remaining_counts_tuple: tuple) -> int:
    tile_counts = dict(tile_counts_tuple)
    remaining_counts = list(remaining_counts_tuple)
    ukeire = 0
    for i in range(34):
        if remaining_counts[i] <= 0: continue
        tmp = tile_counts.copy()
        tmp[i] = tmp.get(i,0) + 1
        new_shanten = calculate_shanten(tmp)
        if new_shanten < base_shanten: ukeire += remaining_counts[i]
    return ukeire

def calculate_ukeire(tile_counts: Dict[int,int], seen_counts: List[int]=None, remaining_counts: List[int]=None) -> int:
    base_shanten = calculate_shanten(tile_counts)
    if remaining_counts is None: rem = [max(0, 4 - tile_counts.get(i,0)) for i in range(34)]
    else: rem = remaining_counts
    tile_counts_tuple = tuple(sorted(tile_counts.items()))
    rem_tuple = tuple(rem)
    return _calculate_ukeire_cached(tile_counts_tuple, base_shanten, rem_tuple)

def compute_seen_and_remaining(hand_counts: Dict[int,int], melds: Dict[str,List[int]], discards: Dict[str,List[int]], dora_indicators: List[int]) -> Tuple[List[int], List[int]]:
    seen = [0]*34
    for k,v in hand_counts.items():
        if 0 <= k < 34: seen[k] += v
    for p, mlist in (melds or {}).items():
        for t in mlist: 
            if 0 <= t < 34: seen[t] += 1
    for p, dlist in (discards or {}).items():
        for t in dlist: 
            if 0 <= t < 34: seen[t] += 1
    for d in (dora_indicators or []):
        if 0 <= d < 34: seen[d] += 1
    remaining = [max(0, 4 - seen[i]) for i in range(34)]
    return seen, remaining

# =========================
# 3. 安全估值模块 (Danger Model)
# =========================

def _is_number_tile(tile_index: int) -> bool:
    return 0 <= tile_index < 27

def is_suji(tile: int, discards: List[int]) -> bool:
    if not _is_number_tile(tile): return False
    t = tile % 9
    suji_tiles = []
    if t - 2 >= 0: suji_tiles.append(tile - 2)
    if t + 2 <= 8 and tile < 9: suji_tiles.append(tile + 2)
    if 9 <= tile < 18:
        if t - 2 >= 0: suji_tiles.append(9 + t - 2)
        if t + 2 <= 8: suji_tiles.append(9 + t + 2)
    if 18 <= tile < 27:
        if t - 2 >= 0: suji_tiles.append(18 + t - 2)
        if t + 2 <= 8: suji_tiles.append(18 + t + 2)
    for s in suji_tiles:
        if s in discards: return True
    return False

def wall_bonus(tile: int, seen_counts: List[int]) -> float:
    c = seen_counts[tile] if seen_counts is not None and 0 <= tile < len(seen_counts) else 0
    if c >= 4: return -120.0
    if c == 3: return -80.0
    if c == 2: return -20.0
    return 0.0

def tile_base_danger(tile: int) -> float:
    if _is_number_tile(tile): return 150.0
    else: return 80.0

def get_tenpai_prob(turn: int, is_riichi: bool) -> float:
    if is_riichi: return 1.0
    if turn <= 4: return 0.03 
    elif turn <= 7: return 0.15
    elif turn <= 10: return 0.45
    elif turn <= 12: return 0.75
    elif turn <= 15: return 0.90
    else: return 1.00

def danger_to_prob(danger: float) -> float:
    if danger < 0: return 0.0
    return 1.0 - math.exp(-danger / 1200.0)

def calculate_danger_score_for_tile(tile_index: int,
                                   seen_counts: List[int],
                                   all_discards: Dict[str, List[int]],
                                   opponents_info: Dict[str, Dict[str, Any]],
                                   turn: int,
                                   dora_tiles: List[int] = None) -> int:
    total_danger = 0.0
    opp_count = 0
    
    for pid, info in opponents_info.items():
        discards = all_discards.get(pid, []) if all_discards is not None else []
        is_riichi = bool(info.get("riichi", False))
        melds = int(info.get("melds", 0))

        # 1. 现物判断 (绝对安全)
        if tile_index in discards:
            opp_count += 1
            continue

        # 2. 基础危险度
        base_danger = tile_base_danger(tile_index)
        
        # [宝牌修正] Dora 危险度惩罚 (+50分)
        if dora_tiles and tile_index in dora_tiles:
            base_danger += 50.0

        # 3. 牌理修正 (筋、壁)
        if is_suji(tile_index, discards):
            base_danger -= 80.0
        base_danger += wall_bonus(tile_index, seen_counts)
        base_danger = max(base_danger, 0.0)

        # 4. 状态修正
        if melds == 1: base_danger *= 1.1
        elif melds == 2: base_danger *= 1.3
        elif melds >= 3: base_danger *= 1.5
        if is_riichi: base_danger *= 1.8

        # 5. 巡目听牌概率修正
        tenpai_prob = get_tenpai_prob(turn, is_riichi)
        final_opp_danger = base_danger * tenpai_prob

        total_danger += final_opp_danger
        opp_count += 1

    if opp_count == 0: return 0
    final_score = total_danger / opp_count
    return int(min(max(final_score, 0), 3000))

def calculate_safe_score_estimate(tile_counts: Dict[int,int],
                                  dora_tiles: List[int],
                                  player_discards: Dict[str,List[int]],
                                  seen_counts: List[int],
                                  remaining_counts: List[int],
                                  pov_player_id: int,
                                  opponents_: Dict[str, Any]=None,
                                  current_turn: int = 1) -> Dict[str, float]:
    
    opponents_info = {}
    if opponents_ is None:
        opponents_info = {f"P{i}": {"riichi": False, "melds": 0} for i in range(4)}
    else:
        example_val = next(iter(opponents_.values())) if opponents_ else None
        if isinstance(example_val, dict):
            opponents_info = opponents_
        else:
            for pid, val in opponents_.items():
                try: is_r = bool(int(val))
                except: is_r = bool(val)
                opponents_info[pid] = {"riichi": is_r, "melds": 0}

    turn = current_turn
    
    dangers = []
    for tile, cnt in tile_counts.items():
        if cnt <= 0: continue
        # 传入 dora_tiles
        d = calculate_danger_score_for_tile(
            tile,
            seen_counts,
            player_discards,
            opponents_info,
            turn,
            dora_tiles=dora_tiles
        )
        dangers.append(d)

    if not dangers:
        return {"safe_prob": 0.0, "avg_prob": 0.0, "risk_prob": 0.0}

    min_danger = min(dangers)
    max_danger = max(dangers)
    avg_danger = sum(dangers) / len(dangers)

    return {
        "safe_prob": danger_to_prob(min_danger),
        "avg_prob": danger_to_prob(avg_danger),
        "risk_prob": danger_to_prob(max_danger)
    }

# =========================
# 4. 格式化与 Prompt 生成
# =========================

def format_single_json_data_direct(data):
    try:
        game_id = data.get('game_id', 'unknown')
        meta = data.get('meta', {})
        scores = data.get('scores', {})
        steps = data.get('steps', [])

        if not steps: return None

        step = steps[0]
        derived_data = step.get('derived', {})

        def format_tile(tile_int):
            if not isinstance(tile_int, int): return str(tile_int)
            if 0 <= tile_int <= 8: return f"{tile_int + 1}万"
            elif 9 <= tile_int <= 17: return f"{tile_int - 8}筒"
            elif 18 <= tile_int <= 26: return f"{tile_int - 17}索"
            elif tile_int == 27: return "东"
            elif tile_int == 28: return "南"
            elif tile_int == 29: return "西"
            elif tile_int == 30: return "北"
            elif tile_int == 31: return "白"
            elif tile_int == 32: return "发"
            elif tile_int == 33: return "中"
            else: return f"未知牌({tile_int})"

        wind_map_str = {"东": 0, "南": 1, "西": 2, "北": 3}
        num_map_zh = ["一", "二", "三", "四"]

        round_wind_val = meta.get('round_wind')
        dealer_idx = meta.get('dealer')
        if isinstance(round_wind_val, str) and round_wind_val in wind_map_str:
            round_wind_str = round_wind_val
        elif isinstance(round_wind_val, int) and 0 <= round_wind_val <= 3:
            round_wind_str = convert_round_wind_to_name(round_wind_val)
        else:
            round_wind_str = "未知"

        if round_wind_str in wind_map_str and isinstance(dealer_idx, int) and 0 <= dealer_idx < 4:
            round_str = f"{round_wind_str}{num_map_zh[dealer_idx]}局"
        else:
            round_str = f"局名解析异常(风={round_wind_str}, 庄={dealer_idx})"

        honba = meta.get('honba', 0)
        wall_tiles = meta.get('wall_tiles', '?')
        richi_sticks = meta.get('richi_sticks', 0)
        turn_num = meta.get('turn', 1)
        
        game_state_str = f"第{turn_num}巡，牌墙余{wall_tiles}张"
        stick_info = ""
        if honba > 0 or richi_sticks > 0:
            stick_info = f"，{honba}本场"
            if richi_sticks > 0:
                stick_info += f"，供托{richi_sticks}根"

        pov_id = step.get('pov_player_id')
        is_dealer_str = "你是庄家" if pov_id == dealer_idx else "你是闲家"

        # [新特性] 计算排位
        my_score = scores.get(f"P{pov_id}", 0)
        all_scores_val = sorted([scores.get(f"P{i}", 0) for i in range(4)], reverse=True)
        try: my_rank = all_scores_val.index(my_score) + 1
        except: my_rank = "?"
        diff_to_top = my_score - all_scores_val[0]
        rank_str = f"当前排名 {my_rank}/4 (与一位差 {diff_to_top})"

        player_names = ["本家", "下家", "对家", "上家"]
        player_positions = [pov_id, (pov_id+1)%4, (pov_id+2)%4, (pov_id+3)%4]
        score_descriptions = []
        for i, pos in enumerate(player_positions):
            if i == 0:
                score_descriptions.append(f"你有 {scores.get(f'P{pos}', '?')}分")
            else:
                score_descriptions.append(f"{player_names[i]}: {scores.get(f'P{pos}', '?')}分")
        scores_str = ", ".join(score_descriptions)

        pov_hand_str = " ".join(step.get('pov_hand', []))
        
        # [修正] 显示实际宝牌而非指示牌
        real_dora_tiles = meta.get('dora_tiles', [])
        dora_str = " ".join([format_tile(t) for t in real_dora_tiles])
        if not dora_str: dora_str = "无"

        melds_info = derived_data.get('melds', {})
        meld_descriptions = []
        for i in range(4):
            pos = (pov_id + i) % 4
            player_melds = melds_info.get(f'P{pos}', [])
            player_name = player_names[i]
            if player_melds:
                meld_descriptions.append(f"{player_name}副露：{' '.join(player_melds)}")
            else:
                meld_descriptions.append(f"{player_name}副露：无")

        all_player_discards = step.get('all_player_discards', {})
        discard_descriptions = []
        for i in range(4):
            pos = (pov_id + i) % 4
            player_discards = all_player_discards.get(f'P{pos}', [])
            player_name = player_names[i]
            if player_discards:
                discard_descriptions.append(f"{player_name}：{' '.join([convert_tile_index_to_name(t) for t in player_discards])}")
            else:
                discard_descriptions.append(f"{player_name}：无")

        safe_est = derived_data.get('safe_score_estimate')
        if isinstance(safe_est, dict):
            # [修正] 防御字段格式化为多行，无显性安牌推荐
            safe_desc = (
                f"\n  最安全牌放铳率：{safe_est.get('safe_prob', 0):.1%}\n"
                f"  平均放铳率：{safe_est.get('avg_prob', 0):.1%}\n"
                f"  最危险牌放铳率：{safe_est.get('risk_prob', 0):.1%}"
            )
        else:
            safe_desc = "未知"

        user_prompt = f"""[情景分析]
- 牌局: {round_str}，{is_dealer_str} ({game_state_str}{stick_info})。
- 状态: {rank_str}。
- 宝牌: {dora_str}
- 各玩家分数: {scores_str}。
- 你的手牌: {pov_hand_str}
- 牌效: {derived_data.get('shanten', '?')} 向听，进张 {derived_data.get('ukeire', '?')} 张。
- 防御：{safe_desc}
场上已见牌信息
各玩家副露信息:{', '.join(meld_descriptions)}
各玩家牌河信息:{', '.join(discard_descriptions)}

[任务]
根据当前情景，选择一张最应该打出的手牌。"""

        assistant_response = step.get('discarded_tile', "")
        if not assistant_response: return None

        formatted_text = f"{user_prompt}\n{assistant_response}"
        return json.dumps({"text": formatted_text}, ensure_ascii=False)

    except Exception as e:
        return None

# =========================
# 5. 批处理与列映射
# =========================

COLUMN_MAP = {
    "game_id": "game_id",
    "round_wind": 0,
    "dealer": 1,
    "honba": 3,
    "round_number": 32,
    "scores": {"P0":6,"P1":7,"P2":8,"P3":9},
    "dora_start": 34, "dora_end": 68,
    "pov_hand_start": 68, "pov_hand_end": 102,
    "p0_meld_start": 102, "p0_meld_end":136,
    "p1_meld_start": 136, "p1_meld_end":170,
    "p2_meld_start": 170, "p2_meld_end":204,
    "p3_meld_start": 204, "p3_meld_end":238,
    "p0_discard_start": 374, "p0_discard_end":408,
    "p1_discard_start": 408, "p1_discard_end":442,
    "p2_discard_start": 442, "p2_discard_end":476,
    "p3_discard_start": 476, "p3_discard_end":510,
    "discard_col": 510
}

def _get_col_value(row_vals, col_index_or_name, columns):
    if isinstance(col_index_or_name, str) and columns is not None:
        try: idx = columns.index(col_index_or_name); return row_vals[idx]
        except ValueError: return None
    try: return row_vals[int(col_index_or_name)]
    except Exception: return None

def monitor_cpu_usage(stop_event, cpu_usage_list):
    while not stop_event.is_set():
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_usage_list.append(cpu_percent)
        time.sleep(0.1)

def process_batch_to_jsonl(batch: np.ndarray, batch_index: int, columns: List[str] = None) -> List[str]:
    jsonl_lines = []
    for i in range(len(batch)):
        row = batch[i]
        
        pov_player_id = int(_get_col_value(row, 2, columns)) if pd.notna(_get_col_value(row, 2, columns)) else 0
        
        def read_index_list_in_range(r, start, end):
            lst = []
            for t_idx, cidx in enumerate(range(start, end)):
                v = _get_col_value(r, cidx, columns)
                if pd.notna(v):
                    try:
                        cnt = int(v)
                        for _ in range(cnt): lst.append(t_idx)
                    except Exception: pass
            return lst

        p0_disc = read_index_list_in_range(row, COLUMN_MAP["p0_discard_start"], COLUMN_MAP["p0_discard_end"])
        p1_disc = read_index_list_in_range(row, COLUMN_MAP["p1_discard_start"], COLUMN_MAP["p1_discard_end"])
        p2_disc = read_index_list_in_range(row, COLUMN_MAP["p2_discard_start"], COLUMN_MAP["p2_discard_end"])
        p3_disc = read_index_list_in_range(row, COLUMN_MAP["p3_discard_start"], COLUMN_MAP["p3_discard_end"])
        
        all_player_discards = {"P0": p0_disc, "P1": p1_disc, "P2": p2_disc, "P3": p3_disc}

        # 全局巡目计算
        total_discards = len(p0_disc) + len(p1_disc) + len(p2_disc) + len(p3_disc)
        real_turn = (total_discards // 4) + 1
        if real_turn < 1: real_turn = 1

        game_id_val = row[511] if len(row) > 511 else ""
        step_data = {
            "game_id": str(game_id_val) if pd.notna(game_id_val) and game_id_val != 0 else "",
            "meta": {
                "round_wind": None,
                "dealer": int(_get_col_value(row, COLUMN_MAP["dealer"], columns)) if pd.notna(_get_col_value(row, COLUMN_MAP["dealer"], columns)) else 0,
                "honba": int(_get_col_value(row, COLUMN_MAP["honba"], columns)) if pd.notna(_get_col_value(row, COLUMN_MAP["honba"], columns)) else 0,
                "round_number": int(_get_col_value(row, COLUMN_MAP["round_number"], columns)) if pd.notna(_get_col_value(row, COLUMN_MAP["round_number"], columns)) else 0,
                "dora_indicator": [],
                "wall_tiles": int(_get_col_value(row, 5, columns)) if pd.notna(_get_col_value(row, 5, columns)) else 70,
                "richi_sticks": int(_get_col_value(row, 4, columns)) if pd.notna(_get_col_value(row, 4, columns)) else 0,
                "turn": real_turn
            },
            "scores": {
                "P0": int(_get_col_value(row, COLUMN_MAP["scores"]["P0"], columns)) if pd.notna(_get_col_value(row, COLUMN_MAP["scores"]["P0"], columns)) else 0,
                "P1": int(_get_col_value(row, COLUMN_MAP["scores"]["P1"], columns)) if pd.notna(_get_col_value(row, COLUMN_MAP["scores"]["P1"], columns)) else 0,
                "P2": int(_get_col_value(row, COLUMN_MAP["scores"]["P2"], columns)) if pd.notna(_get_col_value(row, COLUMN_MAP["scores"]["P2"], columns)) else 0,
                "P3": int(_get_col_value(row, COLUMN_MAP["scores"]["P3"], columns)) if pd.notna(_get_col_value(row, COLUMN_MAP["scores"]["P3"], columns)) else 0
            },
            "steps": []
        }

        rv = _get_col_value(row, COLUMN_MAP["round_wind"], columns)
        if pd.notna(rv):
            try: rvi = int(rv); step_data["meta"]["round_wind"] = convert_round_wind_to_name(rvi) if 0 <= rvi <= 3 else f"无效场风({rv})"
            except Exception: step_data["meta"]["round_wind"] = "未知"
        else: step_data["meta"]["round_wind"] = "未知"

        dora_inds = []
        for tile_index, col_idx in enumerate(range(COLUMN_MAP["dora_start"], COLUMN_MAP["dora_end"])):
            val = _get_col_value(row, col_idx, columns)
            if pd.notna(val):
                try:
                    cnt = int(val)
                    for _ in range(cnt): dora_inds.append(tile_index)
                except Exception: pass
        step_data["meta"]["dora_indicator"] = dora_inds
        
        # 计算并存储实际宝牌
        real_dora_tiles = calculate_dora_tiles(dora_inds)
        step_data["meta"]["dora_tiles"] = real_dora_tiles

        riichi_status = {
            "P0": bool(_get_col_value(row, 10, columns)) if pd.notna(_get_col_value(row, 10, columns)) else False,
            "P1": bool(_get_col_value(row, 11, columns)) if pd.notna(_get_col_value(row, 11, columns)) else False,
            "P2": bool(_get_col_value(row, 12, columns)) if pd.notna(_get_col_value(row, 12, columns)) else False,
            "P3": bool(_get_col_value(row, 13, columns)) if pd.notna(_get_col_value(row, 13, columns)) else False
        }

        pov_hand_idxs = []
        for tile_index, col_idx in enumerate(range(COLUMN_MAP["pov_hand_start"], COLUMN_MAP["pov_hand_end"])):
            val = _get_col_value(row, col_idx, columns)
            if pd.notna(val):
                try:
                    cnt = int(val)
                    for _ in range(cnt): pov_hand_idxs.append(tile_index)
                except Exception: pass

        discarded_tile_index = None
        v = _get_col_value(row, 510, columns)
        if pd.notna(v):
            try:
                vi = int(v)
                if 0 <= vi < 34: discarded_tile_index = vi
            except Exception: pass

        p0_melds = read_index_list_in_range(row, COLUMN_MAP["p0_meld_start"], COLUMN_MAP["p0_meld_end"])
        p1_melds = read_index_list_in_range(row, COLUMN_MAP["p1_meld_start"], COLUMN_MAP["p1_meld_end"])
        p2_melds = read_index_list_in_range(row, COLUMN_MAP["p2_meld_start"], COLUMN_MAP["p2_meld_end"])
        p3_melds = read_index_list_in_range(row, COLUMN_MAP["p3_meld_start"], COLUMN_MAP["p3_meld_end"])
        all_player_melds = {"P0": p0_melds, "P1": p1_melds, "P2": p2_melds, "P3": p3_melds}

        hand_counts = {}
        for t in pov_hand_idxs: hand_counts[t] = hand_counts.get(t, 0) + 1

        current_player_melds = all_player_melds.get(f"P{pov_player_id}", [])
        meld_count = len(current_player_melds)

        seen_counts, remaining_counts = compute_seen_and_remaining(hand_counts, all_player_melds, all_player_discards, dora_inds)
        sh = calculate_shanten(hand_counts, meld_count)
        uk = calculate_ukeire(hand_counts, seen_counts, remaining_counts)

        opponents_info = {}
        for pid in ["P0", "P1", "P2", "P3"]:
            ri = riichi_status.get(pid, False)
            try: ri_bool = bool(int(ri))
            except: ri_bool = bool(ri)
            melds_len = len(all_player_melds.get(pid, [])) if all_player_melds.get(pid) is not None else 0
            opponents_info[pid] = {"riichi": ri_bool, "melds": melds_len}

        safe = calculate_safe_score_estimate(
            hand_counts, 
            real_dora_tiles, # 传入实际宝牌列表
            all_player_discards, 
            seen_counts, 
            remaining_counts, 
            pov_player_id, 
            opponents_info,
            current_turn=real_turn 
        )

        action = {
            "pov_player_id": pov_player_id,
            "step_number": 0, 
            "pov_hand": [convert_tile_index_to_name(t) for t in pov_hand_idxs],
            "discarded_tile": convert_tile_index_to_name(discarded_tile_index) if discarded_tile_index is not None else None,
            "all_player_discards": {
                "P0": p0_disc, "P1": p1_disc, "P2": p2_disc, "P3": p3_disc,
            },
            "derived": {
                "melds": {
                    "P0": [convert_tile_index_to_name(t) for t in p0_melds],
                    "P1": [convert_tile_index_to_name(t) for t in p1_melds],
                    "P2": [convert_tile_index_to_name(t) for t in p2_melds],
                    "P3": [convert_tile_index_to_name(t) for t in p3_melds],
                },
                "remaining_counts": convert_counts_to_names(remaining_counts),
                "shanten": sh,
                "ukeire": uk,
                "safe_score_estimate": safe
            }
        }
        step_data["steps"].append(action)

        jsonl_line = format_single_json_data_direct(step_data)
        if jsonl_line is not None:
            jsonl_lines.append(jsonl_line)

    return jsonl_lines

def process_parquet_to_jsonl_parallel(df: pd.DataFrame, output_file: str, num_workers: int = None, max_json_files: int = None, batch_size: int = 2000) -> int:
    if num_workers is None: num_workers = min(os.cpu_count() + 8, 32)
    values = df.values
    columns = list(df.columns) if df is not None else None
    total_rows = len(values)
    if max_json_files is not None: total_rows = min(total_rows, max_json_files)

    print(f"总共需要处理 {total_rows} 个数据项")
    print(f"批处理大小: {batch_size} 行")
    print(f"工作进程数: {num_workers}")

    processed_rows = 0
    batch_count = 0
    batches = []
    for i in range(0, total_rows, batch_size):
        if processed_rows >= total_rows: break
        end_index = min(i + batch_size, total_rows)
        batch = values[i:end_index]
        batches.append((batch, i))
        processed_rows += len(batch)
        batch_count += 1

    print(f"已准备 {batch_count} 个批处理任务")

    stop_event = threading.Event()
    cpu_usage_list = []
    monitor_thread = threading.Thread(target=monitor_cpu_usage, args=(stop_event, cpu_usage_list))
    monitor_thread.start()

    start_time = time.time()
    total_success_count = 0

    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for batch, batch_index in batches:
                future = executor.submit(process_batch_to_jsonl, batch, batch_index, columns)
                futures.append(future)

            print(f"已提交 {len(futures)} 个批处理任务")

            completed_batches = 0
            batch_times = []
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                for future in as_completed(futures):
                    try:
                        batch_start_time = time.time()
                        jsonl_lines = future.result()
                        batch_end_time = time.time()
                        batch_processing_time = batch_end_time - batch_start_time
                        batch_times.append(batch_processing_time)

                        for line in jsonl_lines: f.write(line + '\n')

                        total_success_count += len(jsonl_lines)
                        completed_batches += 1

                        if completed_batches % max(1, len(futures) // 10) == 0:
                            elapsed_time = time.time() - start_time
                            avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
                            current_cpu = psutil.cpu_percent()
                            print(f"已处理 {completed_batches}/{len(futures)} 个批，耗时: {elapsed_time:.2f}秒，平均批处理时间: {avg_batch_time:.2f}秒，当前CPU: {current_cpu:.1f}%")
                    except Exception as e:
                        print(f"处理批时出错: {e}")
    finally:
        stop_event.set()
        monitor_thread.join()

    if batch_times:
        total_elapsed = time.time() - start_time
        avg_batch_time = sum(batch_times) / len(batch_times)
        avg_cpu_usage = sum(cpu_usage_list) / len(cpu_usage_list) if cpu_usage_list else 0
        print(f"处理完成！总时间: {total_elapsed:.2f}秒")
        print(f"平均批处理时间: {avg_batch_time:.2f}秒")
        print(f"平均CPU使用率: {avg_cpu_usage:.1f}%")
        print(f"成功处理了 {total_success_count} 个数据项")

    return total_success_count

def main():
    import sys
    import glob
    from pathlib import Path

    max_json_files = None
    output_file = None
    num_workers = None

    if len(sys.argv) > 1:
        args = sys.argv[1:]
        for arg in args[:]:
            if arg.startswith("--max="):
                max_json_files = int(arg.split("=")[1])
                args.remove(arg)
                break

        if len(args) >= 1:
            input_path = args[0]
            input_path_obj = Path(input_path)
            if input_path_obj.is_file():
                parquet_files = [input_path]
                folder_name = input_path_obj.parent.name
            elif input_path_obj.is_dir():
                parquet_files = list(glob.glob(os.path.join(input_path, "*.parquet")))
                folder_name = input_path_obj.name
            else:
                print(f"错误: 输入路径不存在 - {input_path}")
                sys.exit(1)

            if len(args) >= 2:
                output_file = args[1]
                num_workers = int(args[2]) if len(args) > 2 else None
        else:
            print("使用示例:")
            print("处理单个文件: python script.py input.parquet [output.jsonl] [num_workers]")
            print("处理文件夹: python script.py input_folder/ [output_folder/] [num_workers]")
            sys.exit(1)
    else:
        print("使用示例:")
        print("处理单个文件: python script.py input.parquet [output.jsonl] [num_workers]")
        print("处理文件夹: python script.py input_folder/ [output_folder/] [num_workers]")
        sys.exit(1)

    if len(parquet_files) == 1 and Path(parquet_files[0]).is_file():
        parquet_file_path = parquet_files[0]
        if output_file is None:
            original_filename = Path(parquet_file_path).stem
            output_file = f"{folder_name}-{original_filename}.jsonl"
        elif Path(output_file).is_dir():
            original_filename = Path(parquet_file_path).stem
            output_file = os.path.join(output_file, f"{folder_name}-{original_filename}.jsonl")

        print(f"开始处理文件: {parquet_file_path}")
        print(f"输出文件: {output_file}")
        print(f"最大工作进程数: {num_workers if num_workers else '自动'}")

        df = read_parquet_file(parquet_file_path)
        if df is None: sys.exit(1)

        success_count = process_parquet_to_jsonl_parallel(df, output_file, num_workers, max_json_files, batch_size=2000)
        print(f"总共成功处理了 {success_count} 个数据项")
    else:
        output_folder = output_file if output_file and Path(output_file).is_dir() else "output_jsonl"
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        print(f"开始处理文件夹: {input_path}")
        print(f"输出文件夹: {output_folder}")
        print(f"找到 {len(parquet_files)} 个.parquet文件")
        print(f"最大工作进程数: {num_workers if num_workers else '自动'}")

        total_success_count = 0
        for i, parquet_file_path in enumerate(parquet_files):
            original_filename = Path(parquet_file_path).stem
            output_file_path = os.path.join(output_folder, f"{folder_name}-{original_filename}.jsonl")

            print(f"处理第 {i+1}/{len(parquet_files)} 个文件: {parquet_file_path}")
            print(f"输出到: {output_file_path}")

            df = read_parquet_file(parquet_file_path)
            if df is None:
                print(f"跳过文件: {parquet_file_path} (读取失败)")
                continue

            success_count = process_parquet_to_jsonl_parallel(df, output_file_path, num_workers, max_json_files, batch_size=2000)
            total_success_count += success_count
            print(f"文件 {parquet_file_path} 处理完成，成功处理 {success_count} 个数据项")

        print(f"所有文件处理完成！总共成功处理了 {total_success_count} 个数据项")

if __name__ == "__main__":
    main()