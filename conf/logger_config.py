import logging
import time
import os

log_dir = 'log'

# log
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

# 全局的日志格式设置
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
# 定义一个KB_SERVICE日志记录器
log_all = logging.getLogger("NLP_SERVICE")
# 设置KB_SERVICE 的输出级别
# log_all.setLevel(logging.DEBUG)
# 设置输出的日志文件名，该文件名会明名为启动时的日期
logNameByDay = log_dir + "/"+time.strftime('%Y-%m-%d', time.localtime(time.time())) + ".log"
# 定义日志输出格式
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
# 创建一个handler，用于写入日志文件
fh = logging.FileHandler(logNameByDay, mode='a', encoding='utf8')
# 设置文件记录器的输出级别
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
# 将logger添加到handle里面
log_all.addHandler(fh)


def get_logger_root():
    return logging.getLogger("NLP_SERVICE")
