# Scrapy settings for simpsons_scraper project

BOT_NAME = 'simpsons_scraper'

SPIDER_MODULES = ['simpsons_scraper.spiders']
NEWSPIDER_MODULE = 'simpsons_scraper.spiders'

# Obey robots.txt rules
ROBOTSTXT_OBEY = False

# Configure concurrent requests
CONCURRENT_REQUESTS = 16
CONCURRENT_REQUESTS_PER_DOMAIN = 16
CONCURRENT_REQUESTS_PER_IP = 16

# Configure a delay for requests for the same website
DOWNLOAD_DELAY = 0.5
RANDOMIZE_DOWNLOAD_DELAY = True

# Enable or disable downloader middlewares
DOWNLOADER_MIDDLEWARES = {
   'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
   'scrapy.downloadermiddlewares.retry.RetryMiddleware': 90,
   'scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware': 110,
}

# Configure item pipelines
ITEM_PIPELINES = {
   'simpsons_scraper.pipelines.SimpsonsScraperPipeline': 300,
}

# Enable and configure the AutoThrottle extension
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1
AUTOTHROTTLE_MAX_DELAY = 60
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
AUTOTHROTTLE_DEBUG = False

# Enable showing throttling stats for every response received
AUTOTHROTTLE_DEBUG = False

# Enable and configure HTTP caching
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 0
HTTPCACHE_DIR = 'httpcache'
HTTPCACHE_IGNORE_HTTP_CODES = []
HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.FilesystemCacheStorage'

# Retry settings
RETRY_ENABLED = True
RETRY_TIMES = 3
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429]

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = 'scraping.log'

# User agent
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36' 