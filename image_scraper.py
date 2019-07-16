from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()

queries = [
"smith grind skateboarding",
"nose grind skateboarding",
"crooked grind skateboarding",
'overcrooked grind skateboarding'
"bluntslide skateboarding",
"noseblunt skateboarding",
"boardslide skateboarding",
"lipslide skateboarding",
"noseslide skateboarding",
"tailslide skateboarding"
]

for query in queries:
    arguments = {"keywords": query,
                 "limit": 300,
                 "print_urls": True,
                 "format": "jpg",
                 "size": "medium",
                 "aspect_ratio": "square",
                 "chromedriver": "/Users/louisspencer/Documents/chromedriver"
                 }
    paths = response.download(arguments)
    print()
