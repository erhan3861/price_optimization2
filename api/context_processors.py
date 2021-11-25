from . import data

price = data.price_class1


def ilk_islec(request):
    #price = price_class
    page = "Türkiye"
    return {'price_class': price}
    