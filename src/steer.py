import pigpio

pi = pigpio.pi()


def trai():
    pi.set_servo_pulsewidth(17, 900)


def phai():
    pi.set_servo_pulsewidth(17, 1600)


def giua():
    pi.set_servo_pulsewidth(17, 1300)