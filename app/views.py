from django.shortcuts import render
from django.shortcuts import HttpResponse
from www.settings import BASE_DIR
import os
from app.assess.classify import classify
from app.assess.test_demo import test


upload_path = os.path.join(BASE_DIR, "static/img")


# Create your views here.
def index(request):
    if request.method == "GET":
        return render(request, 'index.html')
    elif request.method == 'POST':
        img = request.FILES.get('img')
        img_url = ""

        if img:
            filename = img.name
            raw_img_url = os.path.join(upload_path, filename).replace("\\", "/")
            # img.save(raw_img_url)
            with open(raw_img_url, 'wb+') as destination:
                for chunk in img.chunks():
                    destination.write(chunk)

            img_url = os.path.join("/static/img", filename).replace("\\", "/")
            print(raw_img_url)
            print(img_url)
        else:
            return HttpResponse("上传失败!")

        classify_data = classify(image_path=raw_img_url)
        print(classify_data)
        scores = test(image_path=raw_img_url)

        return render(request, 'index.html', dict(classify_data=classify_data,
                                                  scores1=scores[0], scores2=scores[1], scores3=scores[2],
                                                  scores4=scores[3], scores5=scores[4], scores6=scores[5],
                                                  img_url=img_url))


