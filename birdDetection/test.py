from argumentParser import Args

myarg = Args("config/config.yaml")
print(myarg.make_gstreamer_pipeline())