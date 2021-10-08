from deepface import DeepFace

backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']
image_path = '/data/mhasan/src/misc/This_Is_Us/this_is_us_s01_recap_15.jpg'

out = DeepFace.analyze(image_path, detector_backend=backends[4])
print(out)