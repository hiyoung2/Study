from machine.car import drive
from machine.tv import watch


drive()
watch()

print("============================")

from machine import car
from machine import tv

car.drive()
tv.watch()

print("============================")


from machine.test.car import drive
from machine.test.tv import watch

drive()
watch()

print("============================")

from machine.test import car
from machine.test import tv

car.drive()
tv.watch()

'''
운전하다
시청하다
============================
운전하다
시청하다
============================
운전하다
시청하다
============================
운전하다
시청하다
'''



from machine import test
from machine import test

print("============================")

test.car.drive()
test.tv.watch()

'''
운전하다
시청하다
============================
운전하다
시청하다
시청하다
============================
운전하다
시청하다
============================
운전하다
시청하다
============================
운전하다
시청하다
============================
운전하다
시청하다   
'''