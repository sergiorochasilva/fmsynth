import tensorflow as tf, pprint

print("TF version:", tf.__version__)
print("GPU físicos :", tf.config.list_physical_devices("GPU"))
print("GPU lógicos :", tf.config.list_logical_devices("GPU"))
print("Built with CUDA?:", tf.test.is_built_with_cuda())
try:
    pprint.pp(tf.sysconfig.get_build_info())
except Exception as e:
    print("build_info indisponível:", e)
