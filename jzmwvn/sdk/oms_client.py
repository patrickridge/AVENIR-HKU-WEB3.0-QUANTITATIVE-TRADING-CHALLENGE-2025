"""
OMS客户端SDK

基于requests库封装的OMS服务客户端，提供完整的API接口调用功能。
从环境变量读取OMS_URL和OMS_ACCESS_TOKEN配置。

使用示例:
    >>> from sdk.oms_client import OmsClient
    >>> client = OmsClient()
    >>> balances = client.get_balance()
    >>> positions = client.get_position()
    >>> result = client.set_target_position(
    ...     instrument_name="BTC-USDT",
    ...     instrument_type="future", 
    ...     target_value=100,
    ...     position_side="LONG"
    ... )
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
import requests


logger = logging.getLogger(__name__)


class OmsError(Exception):
    """OMS SDK异常基类"""
    pass


class AuthenticationError(OmsError):
    """认证错误"""
    pass


class ApiError(OmsError):
    """API调用错误"""
    pass


class RateLimitError(OmsError):
    """速率限制错误"""
    pass


class ConfigurationError(OmsError):
    """配置错误"""
    pass


class OmsClient:
    """
    OMS SDK客户端
    
    从环境变量读取配置:
    - OMS_URL: OMS服务基础URL
    - OMS_ACCESS_TOKEN: Bearer认证令牌
    
    使用示例:
        >>> client = OmsClient()
        >>> balances = client.get_balance()
        >>> positions = client.get_position()
        >>> client.set_target_position("BTC-USDT", "future", 100, "LONG")
    """
    
    def __init__(self, base_url: Optional[str] = None, access_token: Optional[str] = None, timeout: int = 30):
        """
        初始化OMS客户端
        
        Args:
            base_url: OMS服务基础URL，如果不提供则从环境变量OMS_URL读取
            access_token: Bearer认证令牌，如果不提供则从环境变量OMS_ACCESS_TOKEN读取
            timeout: 请求超时时间（秒）
            
        Raises:
            ConfigurationError: 配置缺失或无效
        """

        # 从环境变量或参数获取配置
        self.base_url = (base_url or os.getenv('OMS_URL', '')).rstrip('/')
        self.access_token = access_token or os.getenv('OMS_ACCESS_TOKEN', '')
        self.timeout = timeout
        
        # 验证配置
        if not self.base_url:
            raise ConfigurationError("OMS_URL environment variable or base_url parameter is required")
        if not self.access_token:
            raise ConfigurationError("OMS_ACCESS_TOKEN environment variable or access_token parameter is required")
        
        # 初始化HTTP会话
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        })
        
        logger.info(f"OmsClient initialized with base_url: {self.base_url}")
    
    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, data: Optional[Dict] = None, raw=False) -> Dict[str, Any]:
        """
        发送HTTP请求
        
        Args:
            method: HTTP方法（GET/POST/PUT/DELETE）
            endpoint: API端点
            data: 请求数据
            
        Returns:
            响应数据字典
            
        Raises:
            AuthenticationError: 认证失败
            RateLimitError: 速率限制
            ApiError: 其他API错误
        """
        url = f"{self.base_url}{endpoint}"
        prepared = requests.Request(method=method, url=url, params=params, json=data, headers=self.session.headers).prepare()
        try:
            response = self.session.send(prepared, timeout=self.timeout)
            if raw:
                return response
        
            # 处理响应状态码
            if response.status_code == 401:
                raise AuthenticationError("Authentication failed. Please check your access token.")
            elif response.status_code == 429:
                raise RateLimitError("Rate limit exceeded. Please try again later.")
            elif response.status_code >= 400:
                error_msg = f"API error: {response.status_code}"
                try:
                    error_data = response.json()
                    if 'message' in error_data:
                        error_msg = f"{error_msg} - {error_data['message']}"
                    elif 'error' in error_data:
                        error_msg = f"{error_msg} - {error_data['error']}"
                except:
                    error_msg = f"{error_msg} - {response.text}"
                raise ApiError(error_msg)
            
            # 解析JSON响应
            try:
                return response.json()
            except json.JSONDecodeError:
                raise ApiError(f"Invalid JSON response: {response.text}")
            
        except requests.exceptions.Timeout:
            raise ApiError(f"Request timeout after {self.timeout} seconds")
        except requests.exceptions.ConnectionError as e:
            raise ApiError(f"Connection error: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise ApiError(f"Request error: {str(e)}")

    def set_target_position(self, instrument_name: str, instrument_type: str, 
                          target_value: float, position_side: str) -> Dict[str, Any]:
        """
        设置目标持仓
        
        Args:
            instrument_name: 交易对名称，如 "BTC-USDT"
            instrument_type: 交易类型，"future" 或 "spot"
            target_value: 目标持仓价值(USDT)
            position_side: 持仓方向，"LONG" 或 "SHORT"
            
        Returns:
            设置结果响应字典，包含任务ID等信息
            
        Raises:
            ApiError: API调用失败
            RateLimitError: 操作过于频繁
        """
        data = {
            "instrument_name": instrument_name,
            "instrument_type": instrument_type,
            "target_value": target_value,
            "position_side": position_side
        }

        try:
            response = self._make_request('POST', '/api/binance/set-target-position', data=data)
            
            if response.get('status') != 'success':
                error_msg = response.get('message', 'Unknown error')
                raise ApiError(f"Failed to set target position: {error_msg}")
            
            logger.info(f"Target position set successfully for {instrument_name}")
            return response['message']
            
        except Exception as e:
            logger.error(f"Error setting target position for {instrument_name}: {str(e)}")
            raise

    def set_target_position_batch(self, elements: List) -> Dict[str, Any]:
        """
        批量设置目标持仓

        Args:
            elements: 包含多个目标持仓设置的字典列表，每个字典包含以下字段：
                - instrument_name: 交易对名称，如 "BTC-USDT"
                - instrument_type: 交易类型，"future" 或 "spot"
                - target_value: 目标持仓价值(USDT)
                - position_side: 持仓方向，"LONG" 或 "SHORT"

        Returns:
            设置结果响应字典，包含任务ID等信息

        Raises:
            ApiError: API调用失败
            RateLimitError: 操作过于频繁
        """

        try:
            response = self._make_request('POST', '/api/binance/set-target-position-batch', data=elements)

            if response.get('status') != 'success':
                error_msg = response.get('message', 'Unknown error')
                raise ApiError(f"Failed to set target position batch: {error_msg}")

            logger.info(f"Target position batch set successfully")
            return response['message']

        except Exception as e:
            logger.error(f"Error setting target position batch: {str(e)}")
            raise

    def get_position(self) -> List[Dict[str, Any]]:
        """
        获取用户持仓列表
        
        Returns:
            持仓信息列表，每个元素包含交易对、持仓方向、数量、���值等信息
            
        Raises:
            ApiError: API调用失败
        """
        try:
            response = self._make_request('GET', '/api/binance/get-position')
            
            if response.get('status') != 'success':
                error_msg = response.get('message', 'Unknown error')
                raise ApiError(f"Failed to get positions: {error_msg}")
            
            positions = response['message']
            logger.info(f"Retrieved {len(positions)} position records")
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            raise

    def get_balance(self) -> List[Dict[str, Any]]:
        """
        获取用户资产列表
        
        Returns:
            资产信息列表，每个元素包含资产类型、余额等信息
            
        Raises:
            ApiError: API调用失败
        """
        try:
            response = self._make_request('GET', '/api/binance/get-balance')
            
            if response.get('status') != 'success':
                error_msg = response.get('message', 'Unknown error')
                raise ApiError(f"Failed to get balances: {error_msg}")
            
            balances = response['message']
            logger.info(f"Retrieved {len(balances)} balance records")
            return balances
            
        except Exception as e:
            logger.error(f"Error getting balances: {str(e)}")
            raise

    def get_asset_changes(self) -> List[Dict[str, Any]]:
        """
        获取用户资产变更历史
        
        Returns:
            资产变更记录列表，包含最近100条变更记录
            
        Raises:
            ApiError: API调用失败
        """
        try:
            response = self._make_request('GET', '/api/binance/get-asset-changes')
            
            if response.get('status') != 'success':
                error_msg = response.get('message', 'Unknown error')
                raise ApiError(f"Failed to get asset changes: {error_msg}")
            
            changes = response['message']
            logger.info(f"Retrieved {len(changes)} asset change records")
            return changes
            
        except Exception as e:
            logger.error(f"Error getting asset changes: {str(e)}")
            raise

    def get_symbols(self) -> List[str]:
        """
        获取可交易合约列表
        
        Returns:
            可交易合约列表

        Raises:
            ApiError: API调用失败
        """
        try:
            response = self._make_request('GET', '/api/market/symbols')
            
            if response.get('status') != 'success':
                error_msg = response.get('message', 'Unknown error')
                raise ApiError(f"Failed to get symbols: {error_msg}")

            symbols = response['symbols']
            logger.info(f"Retrieved {len(symbols)} symbol records")
            return symbols
            
        except Exception as e:
            logger.error(f"Error getting symbols: {str(e)}")
            raise

    def create_strategy_user(self, name: str) -> Dict[str, str]:
        """
        创建策略用户（需要qb-backend权限）
        
        Args:
            name: 用户名
            
        Returns:
            包含用户名和token的字典
            
        Raises:
            ApiError: API调用失败
            AuthenticationError: 权限不足
        """
        data = {"name": name}
        
        try:
            response = self._make_request('POST', '/api/strategy', data)
            
            if 'error' in response:
                raise ApiError(f"Failed to create strategy user: {response['error']}")
            
            logger.info(f"Strategy user '{name}' created successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error creating strategy user '{name}': {str(e)}")
            raise


    def close(self):
        """关闭客户端连接"""
        if self.session:
            self.session.close()
        logger.info("OmsClient closed")

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()


# 便捷函数
def create_client(base_url: Optional[str] = None, access_token: Optional[str] = None) -> OmsClient:
    """
    创建OMS客户端实例
    
    Args:
        base_url: OMS服务基础URL，如果不提供则从环境变量OMS_URL读取
        access_token: Bearer认证令牌，如果不提供则从环境变量OMS_ACCESS_TOKEN读取
        
    Returns:
        OmsClient实例
    """
    return OmsClient(base_url=base_url, access_token=access_token)


# 示例用法
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 创建客户端（从环境变量读取配置）
        client = OmsClient()
        
        # 获取账户信息
        print("=== 获取资产余额 ===")
        balances = client.get_balance()
        for balance in balances:
            print(f"资产: {balance['asset']}, 余额: {balance['balance']}")
        
        print("\n=== 获取持仓信息 ===")
        positions = client.get_position()
        for position in positions:
            print(f"交易对: {position['instrument_name']}, "
                  f"方向: {position['position_side']}, "
                  f"数量: {position['quantity']}, "
                  f"价值: {position['value']}")
        
        print("\n=== 获取资产变更历史 ===")
        changes = client.get_asset_changes()
        for change in changes[:5]:  # 只显示前5条
            print(f"资产: {change['asset']}, "
                  f"变动: {change['change']}, "
                  f"余额: {change['balance']}, "
                  f"时间: {change['create_time']}")
        
        # 设置目标仓位示例（注释掉避免意外执行）
        print("\n=== 设置目标仓位 ===")
        result = client.set_target_position(
            instrument_name="BTC-USDT",
            instrument_type="future",
            target_value=100,
            position_side="LONG"
        )
        print(f"任务ID: {result['id']}")
        
    except ConfigurationError as e:
        print(f"配置错误: {e}")
        print("请设置环境变量 OMS_URL 和 OMS_ACCESS_TOKEN")
    except Exception as e:
        print(f"错误: {e}")
    
    import time
    while True:
        time.sleep(1)
