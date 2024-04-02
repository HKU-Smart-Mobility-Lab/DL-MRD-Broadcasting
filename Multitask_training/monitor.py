from smtplib import SMTP_SSL
from email.mime.text import MIMEText

#邮箱服务器地址，这里我们用的时qq的。要换成163的话这里需要更换。并且如果换成163的话端口号也不一样
mail_host = "smtp.qq.com"
#邮箱登录名
mail_user = '813947820@qq.com'
#密码(部分邮箱为授权码) 
mail_pass = 'adqegpumivcjbdja'
#邮件发送方邮箱地址
sender = '813947820@qq.com'
#接收邮箱的地址
receivers = ['ctj21@connect.hku.hk','shenzj@connect.hku.hk']

message = MIMEText('起床干活了,CPU代码跑完了','plain','utf-8')
#邮件主题       
message['Subject'] = 'CPU Server 代码跑完了'
#发送方信息
message['From'] = sender
#接受方信息     

import os
import time
def autohalt():
    while True:
        ps_string_1 = os.popen('ps ax | grep 2691800','r').read() # 这里的6666是进程号，后面简单说一下怎么查询
        ps_strings_1 = ps_string_1.strip().split('\n')
        # print(ps_strings)
        if len(ps_strings_1)<=2:
            for receiver in receivers:
                message['To'] = receiver
                smtp = SMTP_SSL(mail_host)
                smtp.login(mail_user, mail_pass)
                smtp.sendmail(sender, receivers, message.as_string())
                smtp.quit()
                print('success')
            return
        else:
            # print('Still',len(ps_strings),'Processes, waiting 60s...')
            time.sleep(60) #一分钟后检查一次
if __name__=='__main__':
    autohalt()
