class ModelConfig:
    def __init__(self, model_name):
        self.model_name = model_name
        self.api_base = "https://api.openai.com/v1"
        self.api_key = ""
        self.temperature = 0.7
        self.max_tokens = 1000
        self.top_p = 1.0
        self.frequency_penalty = 0.0
        self.presence_penalty = 0.0
        self.system_message = ""  # 添加System Message字段
        self.request_model = model_name  # 添加实际请求model字段
        self.timeout = 60.0  # 请求超时时间（秒）
        # 并发请求设置
        self.enable_concurrent = False  # 是否开启并发
        self.max_concurrent_requests = 10  # 每分钟最大并发请求数
        self.max_concurrent_total = -1  # 同时并发总数限制（-1为不限制）
        
    def to_dict(self):
        return {
            "model": self.model_name,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "system_message": self.system_message,
            "request_model": self.request_model,
            "timeout": self.timeout,
            "enable_concurrent": self.enable_concurrent,
            "max_concurrent_requests": self.max_concurrent_requests,
            "max_concurrent_total": self.max_concurrent_total
        }
        
    @classmethod
    def from_dict(cls, data):
        # 修复：使用 "model" 而不是 "model_name"
        model_name = data.get("model", data.get("model_name", "unknown"))
        config = cls(model_name)
        config.api_base = data.get("api_base", "https://api.openai.com/v1")
        config.api_key = data.get("api_key", "")
        config.temperature = data.get("temperature", 0.7)
        config.max_tokens = data.get("max_tokens", 1000)
        config.top_p = data.get("top_p", 1.0)
        config.frequency_penalty = data.get("frequency_penalty", 0.0)
        config.presence_penalty = data.get("presence_penalty", 0.0)
        config.system_message = data.get("system_message", "")  # 添加System Message
        config.request_model = data.get("request_model", config.model_name)  # 添加请求model
        config.timeout = data.get("timeout", 60.0)  # 请求超时时间（秒）
        # 并发请求设置
        config.enable_concurrent = data.get("enable_concurrent", False)
        config.max_concurrent_requests = data.get("max_concurrent_requests", 10)
        config.max_concurrent_total = data.get("max_concurrent_total", -1)  # 同时并发总数限制（-1为不限制）
        return config
