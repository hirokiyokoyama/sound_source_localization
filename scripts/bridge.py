#!/usr/bin/env python

import websocket
import json

TOPICS = [('/laser_2d_pose', 'geometry_msgs/PoseWithCovarianceStamped')]
SERVICES = [('/synchronize', 'sound_source_localization/Synchronize')]
PREFIX = '/bridged'

def subscribe_message(topic, typ):
    return '{{\n"op": "subscribe",\n"topic": "{}",\n"type": "{}"\n}}'.format(topic, typ)
def advertise_message(topic, typ):
    return '{{\n"op": "advertise",\n"topic": "{}",\n"type": "{}"\n}}'.format(topic, typ)
def advertise_service_message(service, typ):
    return '{{\n"op": "advertise_service",\n"type": "{}",\n"service": "{}"\n}}'.format(typ, service)
def append_prefix(s):
    return PREFIX + s
def remove_prefix(s):
    assert s.startswith(PREFIX)
    return s[len(PREFIX):]
 
if __name__ == "__main__":
    import rospy
    rospy.init_node('ssl_bridge')
    remote_hostname = rospy.get_param('~remote_robot_hostname')
    websocket.enableTrace(False)

    while not rospy.is_shutdown():
        try:
            ws_local = websocket.create_connection("ws://localhost:9090")
            break
        except:
            rospy.loginfo('Waiting for rosbridge.')
            rospy.sleep(1.)
        
    for topic, typ in TOPICS:
        ws_local.send(advertise_message(append_prefix(topic), typ))
        
    def on_open(ws):
        for topic, typ in TOPICS:
            ws.send(subscribe_message(topic, typ))
        for service, typ in SERVICES:
            rospy.wait_for_service(service)
            ws.send(advertise_service_message(append_prefix(service), typ))
    def on_message(ws, message):
        message = json.loads(message)
        if message['op'] == 'call_service':
            message["service"] = remove_prefix(message["service"])
            ws_local.send(json.dumps(message))
            msg = json.loads(ws_local.recv())
            msg["service"] = append_prefix(msg["service"])
            ws.send(json.dumps(msg))
        elif message['op'] == 'publish':
            message["topic"] = append_prefix(message["topic"])
            ws_local.send(json.dumps(message))
    ws_remote = websocket.WebSocketApp("ws://{}:9090".format(remote_hostname),
                                       on_open = on_open,
                                       on_message = on_message)
    while not rospy.is_shutdown():
        ws_remote.run_forever()

