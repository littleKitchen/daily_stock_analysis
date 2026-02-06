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
    # 传统财经新闻
    "A股 利好 今日",
    "A股 重大合同 公告",
    "上市公司 业绩预增",
    "机构调研 热门股",
    "北向资金 买入",
    "涨停 复盘 龙头",
    # 淘股吧/股吧讨论
    "site:tgb.cn 龙头 涨停",
    "site:guba.eastmoney.com 利好 主力",
    # 雪球讨论
    "site:xueqiu.com 重仓 看好",
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
            self._search_service = SearchService(
                bocha_keys=getattr(self.config, 'bocha_api_keys', None),
                tavily_keys=getattr(self.config, 'tavily_api_keys', None),
                brave_keys=getattr(self.config, 'brave_api_keys', None),
                serpapi_keys=getattr(self.config, 'serpapi_keys', None),
            )
        return self._search_service
    
    @property
    def analyzer(self):
        """懒加载 AI 分析器"""
        if self._analyzer is None:
            from src.analyzer import GeminiAnalyzer
            api_key = getattr(self.config, 'gemini_api_key', None)
            self._analyzer = GeminiAnalyzer(api_key=api_key)
        return self._analyzer
    
    def _generate_content(self, prompt: str) -> Optional[str]:
        """调用 LLM 生成内容"""
        try:
            # 检查 analyzer 是否可用
            if not self.analyzer.is_available():
                logger.warning("AI 分析器不可用")
                return None
            
            # 使用 analyzer 的内部方法调用 API
            generation_config = {
                "temperature": 0.7,
                "max_output_tokens": 2048,
            }
            result = self.analyzer._call_api_with_retry(prompt, generation_config)
            logger.debug(f"LLM 响应长度: {len(result) if result else 0}")
            return result
        except Exception as e:
            logger.warning(f"LLM 调用失败: {type(e).__name__}: {e}")
            return None
    
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
        # 1. 搜索新闻（使用第一个可用的 provider）
        response = None
        for provider in self.search_service._providers:
            if provider.is_available:
                try:
                    response = provider.search(query, max_results=5, days=3)
                    if response.success and response.results:
                        break
                except Exception as e:
                    logger.debug(f"Provider {provider.__class__.__name__} 搜索失败: {e}")
                    continue
        
        if not response or not response.success or not response.results:
            logger.debug(f"搜索 '{query}' 无结果")
            return []
        
        logger.debug(f"搜索 '{query}' 获得 {len(response.results)} 条结果")
        
        # 2. 组装新闻内容
        news_content = self._format_news_for_llm(response.results)
        
        # 3. 调用 LLM 提取
        try:
            signals = self._extract_stocks_from_news(news_content, response.results)
            return signals
        except Exception as e:
            logger.warning(f"LLM 提取股票失败 for '{query}': {type(e).__name__}: {e}")
            return []
    
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
            response = self._generate_content(prompt)
            if not response:
                logger.debug("LLM 返回空响应")
                return []
            
            # 提取 JSON - 尝试找到包含 "stocks" 的 JSON 对象
            # 先尝试找 ```json ... ``` 代码块
            code_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response)
            if code_block_match:
                json_str = code_block_match.group(1)
            else:
                # 否则找最外层的 { }
                json_match = re.search(r'\{[^{}]*"stocks"[^{}]*\[[\s\S]*?\]\s*\}', response)
                if json_match:
                    json_str = json_match.group()
                else:
                    # 最后尝试任何 JSON 对象
                    json_match = re.search(r'\{[\s\S]*\}', response)
                    if not json_match:
                        logger.debug(f"无法从 LLM 响应中提取 JSON: {response[:200]}...")
                        return []
                    json_str = json_match.group()
            
            data = json.loads(json_str)
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
    
    def screen_from_guba(self, top_n: int = 10) -> List[StockSignal]:
        """
        从东方财富股吧热帖中筛选股票
        
        备用数据源，当搜索服务不可用时使用
        """
        import requests
        from bs4 import BeautifulSoup
        
        signals = []
        seen_codes = set()
        
        try:
            # 获取股吧热门帖子页面
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "https://guba.eastmoney.com/",
            }
            
            urls = [
                "https://guba.eastmoney.com/",  # 首页热帖
                "https://guba.eastmoney.com/rank/",  # 人气榜
            ]
            
            for url in urls:
                try:
                    resp = requests.get(url, headers=headers, timeout=10)
                    resp.encoding = 'utf-8'
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    
                    # 提取帖子标题和链接
                    for link in soup.find_all('a', href=True):
                        href = link.get('href', '')
                        title = link.get_text(strip=True)
                        
                        # 股吧帖子链接格式: /news,股票代码,xxx.html
                        if '/news,' in href and title:
                            parts = href.split(',')
                            if len(parts) >= 2:
                                code = parts[1]
                                if self._is_valid_stock_code(code) and code not in seen_codes:
                                    seen_codes.add(code)
                                    signals.append(StockSignal(
                                        code=code,
                                        name="",  # 稍后可通过 API 获取名称
                                        signal_type=SignalType.NEUTRAL,
                                        reason=title[:100],
                                        source="东财股吧",
                                        confidence=0.5,
                                        news_title=title,
                                    ))
                except Exception as e:
                    logger.debug(f"获取 {url} 失败: {e}")
                    continue
            
            logger.info(f"📊 股吧热帖发现 {len(signals)} 只股票")
            
        except Exception as e:
            logger.warning(f"股吧数据获取失败: {e}")
        
        return signals[:top_n]
    
    def screen_combined(self, top_n: int = 10) -> List[StockSignal]:
        """
        综合选股：结合新闻 + 股吧讨论
        
        优先级：新闻利好 > 股吧热议 > 新闻中性
        """
        # 1. 从新闻获取
        news_signals = self.screen_from_news(top_n=top_n * 2)
        
        # 2. 从股吧获取（备用）
        guba_signals = []
        try:
            guba_signals = self.screen_from_guba(top_n=top_n)
        except Exception as e:
            logger.debug(f"股吧数据获取失败，跳过: {e}")
        
        # 3. 合并去重
        seen_codes = set()
        combined = []
        
        # 先加新闻利好
        for s in news_signals:
            if s.code not in seen_codes and s.signal_type == SignalType.POSITIVE:
                combined.append(s)
                seen_codes.add(s.code)
        
        # 再加股吧热议
        for s in guba_signals:
            if s.code not in seen_codes:
                combined.append(s)
                seen_codes.add(s.code)
        
        # 最后补充新闻中性
        for s in news_signals:
            if s.code not in seen_codes:
                combined.append(s)
                seen_codes.add(s.code)
        
        return combined[:top_n]


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
