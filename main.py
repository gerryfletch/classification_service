from classification.datasource import DataSource
from classification.net import Network
from classification.word_hierarchy import Hierarchy

data = [
    # 3 retrievers
    {'description': "golden retriever puppy",
        'ds': DataSource("https://i.pinimg.com/originals/99/f9/ed/99f9ede31328c8484e9e252d08811535.jpg")},
    {'description': "golden retriever adult",
        'ds': DataSource("https://www.prestigeanimalhospital.com/sites/default/files/styles/large/adaptive-image/public/golden-retriever-dog-breed-info.jpg?itok=scGfz-nI")},
    {'description': "golden retriever head",
        'ds': DataSource("https://previews.123rf.com/images/osborn/osborn1105/osborn110500001/9481470-golden-retriever-head-shoulders.jpg")},

    # 3/4 terriers, except toy terrer is not actually a terrier
    {'description': "retriever and terrier",
        'ds': DataSource("https://c8.alamy.com/comp/AJNJT5/labrador-retriever-and-yorkshire-terrier-AJNJT5.jpg")},
    {'description': "yorkshire terrier", 'ds': DataSource(
        "https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2018/03/26125843/yorkshire-terrier-cover-500x486.jpg")},
    {'description': "long haired yorkshire terrier", 'ds': DataSource(
        "https://vetstreet.brightspotcdn.com/dims4/default/851240c/2147483647/thumbnail/645x380/quality/90/?url=https%3A%2F%2Fvetstreet-brightspot.s3.amazonaws.com%2Fdd%2F5c%2Fb14d4033429eaf1ebcd52f8a9d10%2Fyorkshire-terrier-ap-0mhxtf-645.jpg")},
    {'description': "toy terrier", 'ds': DataSource(
        "https://cdn.shopify.com/s/files/1/2548/4866/files/basics-of-dog-breed-english-toy-terrier_large.jpg?v=1546565029")},

    # 3 cats
    {'description': "cat in box",
        'ds': DataSource("https://www.cats.org.uk/media/3236/choosing-a-cat.jpg")},
    {'description': "tabby cat", 'ds': DataSource(
        "https://i2.wp.com/consciouscat.net/wp-content/uploads/2017/09/tabby-cat-2-e1504603272898.jpg?fit=550%2C366&ssl=1")},
    {'description': "tabby kitten", 'ds': DataSource(
        "https://www.natureplprints.com/p/729/rf-brown-tabby-kittenage-6-weeks-19113989.jpg")},

    # 1 misc animals
    {'description': "fish", 'ds': DataSource(
        "https://api.time.com/wp-content/uploads/2019/11/fish-with-human-face-tik-tok-video.jpg")},

    # random tractor
    {'description': "tractor", 'ds': DataSource(
        "https://st.mascus.com/imagetilewm/product/mclean/kioti-dk551c-tractor-st5573,141778_1.jpg")},

    # humans
    {'description': "lady in black top", 'ds': DataSource(
        "https://images.unsplash.com/photo-1508214751196-bcfd4ca60f91?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&w=1000&q=80")},

    # three pianos
    {'description': "grand piano", 'ds': DataSource(
        "https://d1aeri3ty3izns.cloudfront.net/media/49/499349/600/preview.jpg")},
    {'description': "tiny piano", 'ds': DataSource(
        "https://www.thomann.de/pics/bdb/332422/12236247_800.jpg")},
    {'description': "piano keys", 'ds': DataSource(
        "https://imgs.classicfm.com/images/7443?width=5376&crop=16_9&signature=3FIrw5vK3aMvskeaq6ax3dbCCNw=")},

    # four drums
    {'description': 'drum kit', 'ds': DataSource(
        "https://www.thomann.de/pics/bdb/482751/14772471_800.jpg")},
    {'description': 'regiment drum', 'ds': DataSource(
        "https://c8.alamy.com/comp/T7XXFE/a-british-marching-drum-for-the-cameronians-scottish-rifles-20th-century-painted-brass-body-with-wooden-frame-and-white-tension-ropes-intact-drumheads-the-front-bears-the-regimental-coat-of-arms-with-numerous-battle-names-height-37-cm-diameter-37-cm-two-earlier-regiments-the-26th-cameronian-regiment-and-the-90th-perthshire-light-infantry-regiment-were-amalgamated-in-1881-to-form-this-regiment-as-a-rifle-regiment-the-cameronians-had-no-colours-but-displayed-their-battle-honours-on-the-regimental-drums-historic-historical-additional-rights-clearance-info-not-available-T7XXFE.jpg")},
    {'description': 'djembe drum', 'ds': DataSource(
        "https://images-na.ssl-images-amazon.com/images/I/91ONQ0gCRcL._SL1500_.jpg")},
    {'description': 'drummer playing drums', 'ds': DataSource(
        "https://upload.wikimedia.org/wikipedia/commons/a/a3/Olaf_Olsen_Piknik_i_Parken_2017_%28163105%29.jpg")},

    # two guitars
    {'description': 'acoustic guitar', 'ds': DataSource(
        "https://www.pmtonline.co.uk/media/catalog/product/cache/1/image/2400x/9df78eab33525d08d6e5fb8d27136e95/7/9/79293-304864-eastcoast-d1sce-satin-natural-front.jpg")},
    {'description': 'girl playing guitar', 'ds': DataSource(
        "https://previews.123rf.com/images/sheftsoff/sheftsoff1708/sheftsoff170805882/84081992-girl-sings-and-playing-guitar-musician-with-acoustic-guitar.jpg")},

    # food
    {'description': 'bagels', 'ds': DataSource("https://spicesnflavors.com/wp-content/uploads/2020/01/bagel.jpg")},
    {'description': 'ice cream date', 'ds': DataSource("https://i.pinimg.com/originals/5f/67/74/5f6774af64f417c21fb4279462885c7b.jpg")},
    {'description': 'burger date', 'ds': DataSource("https://res.cloudinary.com/twenty20/private_images/t_watermark-criss-cross-10/v1538372066000/photosp/d32464f9-1757-4fec-ac5e-f70a899aebc5/stock-photo-fast-food-burger-fries-date-junk-food-unhealthy-burgers-vegan-food-hand-in-frame-d32464f9-1757-4fec-ac5e-f70a899aebc5.jpg")},
    {'description': 'burger', 'ds': DataSource("https://www.thespruceeats.com/thmb/l4w6PvMqsz1EjueCAh_foPmYafM=/3456x3456/smart/filters:no_upscale()/garlic-burger-patties-333503-hero-01-e4df660ff27b4e5194fdff6d703a4f83.jpg")},
    {'description': 'fruit plate', 'ds': DataSource("https://www.fergusonplarre.com.au/media/catalog/product/cache/1/image/9df78eab33525d08d6e5fb8d27136e95/f/r/fruit-platter.jpg")}
]


net = Network(accuracy_boundary=0)
h = Hierarchy(0.7)

for i in range(len(data)):
    cfs = net.classify(data[i]['ds'])
    print(f"\n-- {data[i]['description']} --")
    h.place(
        cfs
    )

print("\n\n\n\n")
h.print()
