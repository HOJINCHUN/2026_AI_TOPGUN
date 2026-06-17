# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.sim.state_schema import StateIndex

# 커리큘럼에서 오버라이드 값이 없을 경우 사용될 기본 가중치 세팅
MY_REWARD_CONFIG = {
    "step_penalty": -0.01,
    "pursuit_scale": 1.0,
    "wez_snap_bonus": 10.0,  # Phase 1: 1도 이내 극한의 조준 성공 시 막대한 보상
    "win_reward": 100.0,
    "loss_reward": -100.0,
    "draw_reward": -10.0,
}

def compute_reward(
    ownship_state,
    target_state,
    ownship_damage: float,
    target_damage: float,
    geo_info,
    wez_config: dict,
    reward_config: dict,
    terminated: bool,
    truncated: bool,
    end_condition: str,
) -> tuple[float, dict]:
    
    components: dict[str, float] = {}

    if not (terminated or truncated):
        # 1. 기본 스텝 페널티 (시간 지연 방지)
        components["step"] = float(reward_config.get("step_penalty", -0.01))

        pursuit_scale = float(reward_config.get("pursuit_scale", 1.0))
        if pursuit_scale > 0.0:
            # 거리 계산 (미터 -> 피트 변환)
            dist_m = geo_info._get_distance(ownship_state, target_state)
            dist_ft = dist_m * 3.28084
            
            # 내 기수와 적 사이의 오차 각도(ATA) 계산
            ata = geo_info._get_antenna_train_angle(ownship_state, target_state, False)
            abs_ata = abs(ata)

            # -----------------------------------------------------------
            # [모드 A] BVR (무승부 연장전) 원거리 접근 모드 (> 10,000ft)
            # -----------------------------------------------------------
            if dist_ft > 10000.0:
                # 3만ft에서 0점, 1만ft에 도달할 때 1.0점
                range_score = max(0.0, 1.0 - ((dist_ft - 10000.0) / 20000.0))
                # 정면 180도 반경(90도) 안에만 두면 점수 부여
                angle_score = max(0.0, 1.0 - (abs_ata / 90.0))
                
                # 원거리에서는 거리를 좁히는 것(0.7)이 각도(0.3)보다 중요함
                components["pursuit"] = pursuit_scale * (range_score * 0.7 + angle_score * 0.3)

            # -----------------------------------------------------------
            # [모드 B] WVR (정규전) 도그파이트 모드 (<= 3,000ft)
            # -----------------------------------------------------------
            elif dist_ft <= 3000.0:
                # 🌟 충돌 방지 '스윗 스팟(Sweet Spot)' 거리 계산 🌟
                if dist_ft >= 500.0:
                    # 500ft ~ 3000ft 구간: 500ft에 가까워질수록 1.0점 도달
                    range_score = 1.0 - ((dist_ft - 500.0) / 2500.0)
                else:
                    # 500ft 미만 구간 (충돌 위험): 너무 가까워지면 점수 삭감 (0ft = 0점)
                    range_score = dist_ft / 500.0 

                # 30도 이내로 기수를 좁히도록 유도
                angle_score = max(0.0, 1.0 - (abs_ata / 30.0))
                
                # 근접전에서는 꼬리를 무는 각도(0.8)가 거리(0.2)보다 압도적으로 중요함
                components["pursuit"] = pursuit_scale * (angle_score * 0.8 + range_score * 0.2)

                # 🌟 [Phase 1] 스나이퍼 WEZ 조준 보상 🌟
                # 조건: 500ft <= 거리 <= 3000ft AND 오차 각도 < 1도 (대미지 계수 1 조건)
                if 500.0 <= dist_ft <= 3000.0 and abs_ata < 1.0:
                    components["wez_snap"] = float(reward_config.get("wez_snap_bonus", 10.0))
            
            # -----------------------------------------------------------
            # [모드 C] 전환 구역 (3,000ft ~ 10,000ft)
            # -----------------------------------------------------------
            else:
                # BVR에서 WVR로 넘어가는 중간 단계 (안정적인 접근 유지 보상)
                components["pursuit"] = pursuit_scale * 0.5

    # 2. 터미널 보상 (게임 종료 시 최종 승패 판정)
    terminal_reward = 0.0
    if terminated or truncated:
        ownship_health = float(ownship_state[StateIndex.HEALTH])
        target_health = float(target_state[StateIndex.HEALTH])
        
        if target_health <= 0.0 < ownship_health:
            terminal_reward = float(reward_config.get("win_reward", 100.0))
        elif ownship_health <= 0.0 < target_health:
            terminal_reward = float(reward_config.get("loss_reward", -100.0))
        else:
            terminal_reward = float(reward_config.get("draw_reward", -10.0))
            
    components["terminal"] = terminal_reward

    # components 딕셔너리에 담긴 모든 보상/페널티의 총합 반환
    return float(sum(components.values())), components

__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
