# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - 智能选股模块
===================================

职责：
1. 从财经新闻中自动发现值得关注的股票
2. 利用 LLM 分析新闻情绪和提取股票代码
3. 支持多种选股策略（新闻驱动、技术面筛选、板块轮动）
4. 返回 Top N 股票供后续分析

使用方式：
    from src.stock_screener import StockScreener
    
    screener = StockScreener(config)
    stocks = screener.screen_from_news(top_n=5)
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """信号类型"""
    POSITIVE = "利好"
    NEGATIVE = "利空"
    NEUTRAL = "中性"


@dataclass
class StockSignal:
    """股票信号"""
    code: str           # 股票代码 (如 600519, 300750)
    name: str           # 股票名称
    signal_type: SignalType
    reason: str         # 原因/新闻摘要
    source: str         # 新闻来源
    confidence: float   # 置信度 0-1
    news_title: str     # 新闻标题
    
    def __repr__(self):
        return f"StockSignal({self.code} {self.name} [{self.signal_type.value}] 置信度:{self.confidence:.0%})"


# LLM 提取股票的 Prompt
EXTRACT_STOCKS_PROMPT = """你是一个专业的 A 股分析师。请从以下财经新闻中提取被提及的 A 股股票。

要求：
1. 只提取 A 股股票（上交所、深交所、北交所）
2. 判断每只股票是利好还是利空
3. 给出置信度（0-1，越高越确定）
4. 优先关注：政策利好、业绩超预期、重大合同、并购重组、机构调研等
5. 忽略：纯行情报道、无实质内容的新闻

请用 JSON 格式返回，示例：
```json
{
  "stocks": [
    {
      "code": "600519",
      "name": "贵州茅台",
      "signal": "positive",
      "confidence": 0.85,
      "reason": "公司宣布提价10%，利润预期上调"
    },
    {
      "code": "300750",
      "name": "宁德时代",
      "signal": "negative",
      "confidence": 0.7,
      "reason": "欧盟反补贴调查可能影响出口"
    }
  ]
}
```

如果新闻中没有值得关注的股票，返回空数组：
```json
{"stocks": []}
```

---
新闻内容：
{news_content}
"""

# 搜索财经新闻的查询词
NEWS_QUERIES = [
    "A股 利好 今日",
    "A股 重大合同 公告",
    "上市公司 业绩预增",
    "机构调研 热门股",
    "北向资金 买入",
    "涨停 复盘 龙头",
]


class StockScreener:
    """智能选股器"""
    
    def __init__(self, config, search_service=None, analyzer=None):
        """
        初始化选股器
        
        Args:
            config: 系统配置
            search_service: 搜索服务（可选，不传则自动创建）
            analyzer: AI 分析器（可选，不传则自动创建）
        """
        self.config = config
        self._search_service = search_service
        self._analyzer = analyzer
        
    @property
    def search_service(self):
        """懒加载搜索服务"""
        if self._search_service is None:
            from src.search_service import SearchService
            self._search_service = SearchService(self.config)
        return self._search_service
    
    @property
    def analyzer(self):
        """懒加载 AI 分析器"""
        if self._analyzer is None:
            from src.analyzer import GeminiAnalyzer
            self._analyzer = GeminiAnalyzer(self.config)
        return self._analyzer
    
    def screen_from_news(self, top_n: int = 10, queries: List[str] = None) -> List[StockSignal]:
        """
        从新闻中筛选股票
        
        Args:
            top_n: 返回 Top N 只股票
            queries: 自定义搜索关键词（默认使用内置词）
            
        Returns:
            按置信度排序的股票信号列表
        """
        queries = queries or NEWS_QUERIES
        all_signals: List[StockSignal] = []
        seen_codes = set()
        
        logger.info(f"🔍 开始新闻选股，搜索 {len(queries)} 个关键词...")
        
        for query in queries:
            try:
                signals = self._search_and_extract(query)
                for signal in signals:
                    if signal.code not in seen_codes:
                        all_signals.append(signal)
                        seen_codes.add(signal.code)
                        logger.info(f"  发现: {signal}")
            except Exception as e:
                logger.warning(f"搜索 '{query}' 失败: {e}")
                continue
        
        # 按置信度排序，优先利好
        all_signals.sort(key=lambda x: (
            x.signal_type == SignalType.POSITIVE,  # 利好优先
            x.confidence  # 置信度高优先
        ), reverse=True)
        
        result = all_signals[:top_n]
        logger.info(f"✅ 新闻选股完成，发现 {len(all_signals)} 只，返回 Top {len(result)}")
        
        return result
    
    def _search_and_extract(self, query: str) -> List[StockSignal]:
        """搜索新闻并提取股票"""
        # 1. 搜索新闻
        response = self.search_service.search(query, max_results=5)
        if not response.success or not response.results:
            return []
        
        # 2. 组装新闻内容
        news_content = self._format_news_for_llm(response.results)
        
        # 3. 调用 LLM 提取
        signals = self._extract_stocks_from_news(news_content, response.results)
        
        return signals
    
    def _format_news_for_llm(self, results) -> str:
        """格式化新闻供 LLM 分析"""
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"[新闻{i}] {r.title}\n来源: {r.source}\n摘要: {r.snippet}\n")
        return "\n---\n".join(parts)
    
    def _extract_stocks_from_news(self, news_content: str, results) -> List[StockSignal]:
        """用 LLM 从新闻中提取股票"""
        import json
        
        prompt = EXTRACT_STOCKS_PROMPT.format(news_content=news_content)
        
        try:
            response = self.analyzer.generate_content(prompt)
            if not response:
                return []
            
            # 提取 JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                return []
            
            data = json.loads(json_match.group())
            stocks_data = data.get("stocks", [])
            
            signals = []
            for s in stocks_data:
                code = s.get("code", "").strip()
                if not self._is_valid_stock_code(code):
                    continue
                
                signal_type = {
                    "positive": SignalType.POSITIVE,
                    "negative": SignalType.NEGATIVE,
                }.get(s.get("signal", "").lower(), SignalType.NEUTRAL)
                
                signals.append(StockSignal(
                    code=code,
                    name=s.get("name", "未知"),
                    signal_type=signal_type,
                    reason=s.get("reason", ""),
                    source=results[0].source if results else "新闻",
                    confidence=float(s.get("confidence", 0.5)),
                    news_title=results[0].title if results else "",
                ))
            
            return signals
            
        except Exception as e:
            logger.warning(f"LLM 提取股票失败: {e}")
            return []
    
    def _is_valid_stock_code(self, code: str) -> bool:
        """验证是否是有效的 A 股代码"""
        if not code or not code.isdigit():
            return False
        if len(code) != 6:
            return False
        # 上交所: 60/68, 深交所: 00/30, 北交所: 8/4
        prefix = code[:2]
        return prefix in ("60", "68", "00", "30", "83", "43", "87")
    
    def get_stock_codes(self, top_n: int = 10) -> List[str]:
        """
        便捷方法：直接返回股票代码列表
        
        可直接用于替换 config.stock_list
        """
        signals = self.screen_from_news(top_n=top_n)
        return [s.code for s in signals]


# === 命令行测试 ===
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__).rsplit("/src/", 1)[0])
    
    from src.config import get_config
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s'
    )
    
    config = get_config()
    screener = StockScreener(config)
    
    print("\n" + "="*50)
    print("🔍 智能选股测试")
    print("="*50 + "\n")
    
    signals = screener.screen_from_news(top_n=5)
    
    print("\n📊 选股结果:")
    print("-"*50)
    for i, s in enumerate(signals, 1):
        print(f"{i}. {s.code} {s.name}")
        print(f"   信号: {s.signal_type.value} | 置信度: {s.confidence:.0%}")
        print(f"   原因: {s.reason}")
        print()
