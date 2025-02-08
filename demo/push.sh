# adb root
# adb remount
adb push orc /sdcard/
adb push ./runner /system/bin/
adb push ./libc++_shared.so /system/lib/
adb push ./libDL_api_shared.so /system/lib/
adb push ./libopencv_img_hash.so /system/lib/
adb push ./libopencv_world.so /system/lib/
adb shell runner /sdcard/orc/image.png
adb pull /sdcard/vis.png
