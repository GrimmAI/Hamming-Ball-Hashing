import os
from datetime import datetime


class Log:
    def __init__(self, log_file):
        # 获取文件夹路径和文件名
        self.log_dir = os.path.dirname(log_file)
        self.log_file = log_file

        # 如果指定的文件夹不存在，则创建该文件夹
        if self.log_dir and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # 创建日志文件，如果不存在的话
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("Log File Created: {}\n".format(datetime.now()))

    def write(self, message):
        """将消息写入日志文件和控制台"""
        # 获取当前时间戳
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"{timestamp} - {message}\n"

        # 写入日志文件
        with open(self.log_file, 'a') as f:
            f.write(log_message)

        # 在控制台打印日志消息
        print(log_message, end='')


# 使用示例
if __name__ == "__main__":
    logger = Log('logs/test.log')  # 可以指定日志文件名
    logger.write("This is a log message.")
    logger.write("Another log message.")
