import pika
import socket
import chardet
from datetime import datetime

import ai_verify as AI


class RabbitMQServer:

    def __init__(self):
        self.hostname = socket.gethostbyname(socket.gethostname())
        self.port = 5672
        self.virtual_host = '/'
        self.exchange_name = 'amq.direct'
        self.user_id = 'rabbitmq'
        self.user_pw = 'rabbitmq'
        self.queue_name = 'airepair'
        self.routing_key = 'airepair'

        self.ai_verify = AI.AIVerify()

    def set_ai_verify(self):
        self.ai_verify.load_model()

    def channel_connect(self):
        credential = pika.PlainCredentials(self.user_id, self.user_pw)
        parameters = pika.ConnectionParameters(host=self.hostname,
                                               port=self.port,
                                               virtual_host=self.virtual_host,
                                               credentials=credential,
                                               heartbeat=600)
        connection = pika.BlockingConnection(parameters)

        channel = connection.channel()
        channel.exchange_declare(exchange=self.exchange_name,
                                 exchange_type='direct',
                                 durable=True)
        channel.queue_declare(queue=self.queue_name)
        channel.queue_bind(exchange=self.exchange_name,
                           queue=self.queue_name,
                           routing_key=self.routing_key)

        # 메세지 여러개 쌓여있을 경우 1개씩 끊어서 처리
        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=self.queue_name,
                              on_message_callback=self.callback,
                              auto_ack=True)

        print(f' [#] IP : {self.hostname} | Port : {self.port}')
        print(f' [#] ID : {self.user_id} | PW : {self.user_pw}')
        print('<br> [*] Waiting for messages.<br>')         # html 문법 \n = <br>
        channel.start_consuming()

    def callback(self, ch, method, properties, body):
        msg = body.decode(chardet.detect(body)['encoding'])
        print(' [x] Received %r' % msg)
        self.ai_verify.inspect_ftp(msg)
        # self.ai_verify.inspect_http(msg)
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'{now} : Finish')
        print('<br> [*] Waiting for messages.<br>')         # html 문법 \n = <br>


if __name__ == '__main__':
    server = RabbitMQServer()
    try:
        server.set_ai_verify()
    except Exception as ex:
        print(ex)
    else:
        while True:
            try:
                server.channel_connect()
            except Exception as ex:
                print(ex)
