#!/bin/bash
# ============================================================
# GitHub DNS IP 速度测试脚本 v2
# 兼容 Windows 中文/英文 ping 输出
# 主要用 TCP 连接时间 + Ping 延迟综合评估
# ============================================================

IPS=(
  "140.82.112.3"
  "140.82.113.3"
  "140.82.113.4"
  "140.82.114.3"
  "140.82.114.4"
  "140.82.121.3"
  "140.82.121.4"
  "4.228.31.150"
  "4.237.22.38"
  "20.26.156.215"
  "20.87.245.0"
  "20.200.245.247"
  "20.205.243.166"
  "20.207.73.82"
)

PING_COUNT=4
TIMEOUT=4
TMPFILE=$(mktemp)
trap 'rm -f "$TMPFILE"' EXIT

echo "============================================================"
echo "  GitHub DNS IP 速度测试"
echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Ping: ${PING_COUNT}次  |  TCP: 443端口"
echo "============================================================"
echo ""

# ---------- 从 ping 输出提取平均延迟 (兼容中英文) ----------
extract_ping_avg() {
  local raw="$1"
  local avg

  # 尝试英文: "Average = 249ms"
  avg=$(echo "$raw" | grep -i "Average" | grep -oP '=\s*\K[0-9]+(?=ms)' | head -1)
  if [ -n "$avg" ]; then echo "$avg"; return 0; fi

  # 尝试从统计行提取 — 通常最后一行包含 avg/min/max，avg在第二个 "=" 之后
  # 比如: "Minimum = 248ms, Maximum = 257ms, Average = 251ms"
  # 或者用 sed 抓最后一个 = NNNms
  avg=$(echo "$raw" | grep -oP '=\s*\K[0-9]+(?=ms)' | tail -1)
  if [ -n "$avg" ]; then echo "$avg"; return 0; fi

  # 最后尝试: 找所有 "时间=Nms" 或 "time=Nms" 取平均
  local times
  times=$(echo "$raw" | grep -oP '(?:时间|time)[=<]\s*\K[0-9]+(?=ms)' | head -"$PING_COUNT")
  if [ -n "$times" ]; then
    local sum=0 cnt=0
    for t in $times; do sum=$((sum + t)); cnt=$((cnt + 1)); done
    [ "$cnt" -gt 0 ] && echo $((sum / cnt)) && return 0
  fi

  echo "-1"
  return 1
}

extract_ping_loss() {
  local raw="$1"
  local loss

  # "(0% loss)" or "(0% 丢失)"
  loss=$(echo "$raw" | grep -oP '\(\K[0-9]+(?=%)' | head -1)
  if [ -n "$loss" ]; then echo "$loss"; return 0; fi

  # "100% loss" or "100% 丢失"
  loss=$(echo "$raw" | grep -oP '[0-9]+(?=% packet loss|% loss|% 丢失)' | head -1)
  if [ -n "$loss" ]; then echo "$loss"; return 0; fi

  # 如果 ping 完全失败
  if echo "$raw" | grep -qi "unreachable\|timed out\|could not find"; then
    echo "100"; return 0
  fi

  echo "-1"
}

# ---------- TCP 连接测试 ----------
test_tcp() {
  local ip="$1"
  # 测 3 次取平均
  local total=0 cnt=0 t
  for i in 1 2 3; do
    t=$(curl -s -o /dev/null --connect-timeout "$TIMEOUT" --max-time "$TIMEOUT" \
      -w "%{time_connect}" "https://$ip" 2>/dev/null)
    local ec=$?
    # exit code: 0=ok, 35=SSL handshake fail, 60=SSL cert error — all mean TCP connected
    if [ "$ec" -eq 0 ] || [ "$ec" -eq 35 ] || [ "$ec" -eq 60 ]; then
      total=$(awk "BEGIN {printf \"%.6f\", $total + $t}")
      cnt=$((cnt + 1))
    fi
  done
  if [ "$cnt" -gt 0 ]; then
    awk "BEGIN {printf \"%.4f\", $total / $cnt}"
  else
    echo "FAIL"
  fi
}

# ---------- 主循环 ----------
declare -A P_AVG P_LOSS TCP_TIME
i=1
total=${#IPS[@]}

for ip in "${IPS[@]}"; do
  printf "[%2d/%2d] %-16s " "$i" "$total" "$ip"

  # Ping
  raw_ping=$(ping -n "$PING_COUNT" -w $((TIMEOUT * 1000)) "$ip" 2>&1)
  pavg=$(extract_ping_avg "$raw_ping")
  ploss=$(extract_ping_loss "$raw_ping")

  if [ "$pavg" = "-1" ] || [ "$ploss" = "100" ]; then
    printf "Ping: 不可达 | TCP: "
    P_AVG["$ip"]=9999
    P_LOSS["$ip"]=100
  else
    printf "Ping: %3sms (loss %s%%) | TCP: " "$pavg" "$ploss"
    P_AVG["$ip"]="$pavg"
    P_LOSS["$ip"]="$ploss"
  fi

  # TCP
  tcp_time=$(test_tcp "$ip")
  if [ "$tcp_time" = "FAIL" ]; then
    echo "不可达"
    TCP_TIME["$ip"]=9999
  else
    tcp_ms=$(awk "BEGIN {printf \"%.0f\", $tcp_time * 1000}")
    echo "${tcp_ms}ms"
    TCP_TIME["$ip"]="$tcp_time"
  fi

  ((i++))
done

# ---------- 排名 ----------
echo ""
echo "============================================================"
echo "  综合排名 (Ping 延迟 + TCP 连接时间)"
echo "============================================================"
printf "%-4s %-18s %-8s %-8s %-6s %-8s %-s\n" \
  "排名" "IP" "Ping" "TCP" "丢包" "评分" "评价"
printf "%-4s %-18s %-8s %-8s %-6s %-8s %-s\n" \
  "----" "----------------" "------" "------" "----" "------" "----"

# 计算综合评分 (ping 权重 0.6 + tcp 权重 0.4)
declare -A SCORES
ranked=""
for ip in "${IPS[@]}"; do
  p=${P_AVG["$ip"]}
  t=${TCP_TIME["$ip"]}

  if [ "$p" = "9999" ] && [ "$t" = "9999" ]; then
    SCORES["$ip"]=99999
  elif [ "$p" = "9999" ]; then
    SCORES["$ip"]=$(awk "BEGIN {printf \"%.1f\", $t * 1000}")
  elif [ "$t" = "9999" ]; then
    SCORES["$ip"]=$(awk "BEGIN {printf \"%.1f\", $p * 1.0}")
  else
    # 综合: ping_ms * 0.6 + tcp_ms * 0.4
    tcp_ms=$(awk "BEGIN {printf \"%.0f\", $t * 1000}")
    SCORES["$ip"]=$(awk "BEGIN {printf \"%.1f\", $p * 0.6 + $tcp_ms * 0.4}")
  fi
  ranked+="${SCORES[$ip]}|$ip"$'\n'
done

rank=0
while IFS='|' read -r score ip; do
  [ -z "$ip" ] && continue
  ((rank++))
  p=${P_AVG["$ip"]}
  t=${TCP_TIME["$ip"]}
  loss=${P_LOSS["$ip"]}

  if [ "$p" = "9999" ]; then p_disp="N/A"; else p_disp="${p}ms"; fi
  if [ "$t" = "9999" ]; then t_disp="N/A"; else t_disp="$(awk "BEGIN {printf \"%.0f\", $t * 1000}")ms"; fi

  # 评价
  if [ "$p" = "9999" ] && [ "$t" = "9999" ]; then
    grade="❌ 不可达"
  elif [ "$p" != "9999" ] && [ "$p" -lt 80 ]; then
    grade="⭐ 极快"
  elif [ "$p" != "9999" ] && [ "$p" -lt 180 ]; then
    grade="✅ 快"
  elif [ "$p" != "9999" ] && [ "$p" -lt 300 ]; then
    grade="⚠️  一般"
  else
    grade="🐌 慢"
  fi

  printf "%-4s %-18s %-8s %-8s %-6s %-8s %-s\n" \
    "#${rank}" "$ip" "$p_disp" "$t_disp" "${loss}%" "$score" "$grade"
done < <(echo "$ranked" | sort -t'|' -k1 -n)

# ---------- 推荐 top 3 ----------
echo ""
echo "============================================================"
echo "  🏆 推荐 hosts 配置 (前 3)"
echo "============================================================"
echo ""
rank=0
while IFS='|' read -r score ip; do
  [ -z "$ip" ] && continue
  ((rank++))
  [ $rank -gt 3 ] && break
  p=${P_AVG["$ip"]}
  t=${TCP_TIME["$ip"]}
  echo "# ${rank}: ${ip}  (Ping: ${p}ms, TCP: $(awk "BEGIN {printf \"%.0f\", $t * 1000}")ms)"
  echo "${ip} github.com"
  echo ""
done < <(echo "$ranked" | sort -t'|' -k1 -n)

echo "追加到 C:\\Windows\\System32\\drivers\\etc\\hosts 即可"
echo "测试完成后建议用 ipconfig /flushdns 刷新 DNS 缓存"
echo "============================================================"
