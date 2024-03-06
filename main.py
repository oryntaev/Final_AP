import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the pre-trained TensorFlow model
model = tf.keras.models.load_model('flowers_efficientnet_model.h5')

# Dictionary of flower labels
flower_labels = {
    "21": "fire lily", "3": "canterbury bells", "45": "bolero deep blue", "1": "pink primrose", "34": "mexican aster",
    "27": "prince of wales feathers", "7": "moon orchid", "16": "globe-flower", "25": "grape hyacinth",
    "26": "corn poppy",
    "79": "toad lily", "39": "siam tulip", "24": "red ginger", "67": "spring crocus", "35": "alpine sea holly",
    "32": "garden phlox", "10": "globe thistle", "6": "tiger lily", "93": "ball moss", "33": "love in the mist",
    "9": "monkshood", "102": "blackberry lily", "14": "spear thistle", "19": "balloon flower", "100": "blanket flower",
    "13": "king protea", "49": "oxeye daisy", "15": "yellow iris", "61": "cautleya spicata", "31": "carnation",
    "64": "silverbush", "68": "bearded iris", "63": "black-eyed susan", "69": "windflower", "62": "japanese anemone",
    "20": "giant white arum lily", "38": "great masterwort", "4": "sweet pea", "86": "tree mallow",
    "101": "trumpet creeper", "42": "daffodil", "22": "pincushion flower", "2": "hard-leaved pocket orchid",
    "54": "sunflower", "66": "osteospermum", "70": "tree poppy", "85": "desert-rose", "99": "bromelia",
    "87": "magnolia", "5": "english marigold", "92": "bee balm", "28": "stemless gentian", "97": "mallow",
    "57": "gaura", "40": "lenten rose", "47": "marigold", "59": "orange dahlia", "48": "buttercup",
    "55": "pelargonium", "36": "ruby-lipped cattleya", "91": "hippeastrum", "29": "artichoke", "71": "gazania",
    "90": "canna lily", "18": "peruvian lily", "98": "mexican petunia", "8": "bird of paradise", "30": "sweet william",
    "17": "purple coneflower", "52": "wild pansy", "84": "columbine", "12": "colt's foot", "11": "snapdragon",
    "96": "camellia", "23": "fritillary", "50": "common dandelion", "44": "poinsettia", "53": "primula",
    "72": "azalea", "65": "californian poppy", "80": "anthurium", "76": "morning glory", "37": "cape flower",
    "56": "bishop of llandaff", "60": "pink-yellow dahlia", "82": "clematis", "58": "geranium", "75": "thorn apple",
    "41": "barbeton daisy", "95": "bougainvillea", "43": "sword lily", "83": "hibiscus", "78": "lotus",
    "88": "cyclamen", "94": "foxglove", "81": "frangipani", "74": "rose", "89": "watercress", "73": "water lily",
    "46": "wallflower", "77": "passion flower", "51": "petunia"
}

# Dictionary of flower descriptions
flower_descriptions = {
    "fire lily": "The fire lily, or Gloriosa superba, is a striking flower native to Africa and Asia. It is known for its vibrant red and yellow petals that resemble flames.",
    "canterbury bells": "Canterbury bells, or Campanula medium, are biennial flowering plants native to southern Europe. They produce bell-shaped flowers in various shades of blue, purple, pink, and white.",
    "bolero deep blue": "Bolero deep blue is a variety of delphinium, a tall perennial flower with deeply lobed leaves and spiky blue flowers. It adds vertical interest to garden borders and attracts pollinators.",
    "pink primrose": "The pink primrose, or Oenothera speciosa, is a wildflower native to North America. It has delicate pink petals and blooms in the spring and summer.",
    "mexican aster": "The Mexican aster, or Cosmos bipinnatus, is a popular garden flower native to Mexico. It has daisy-like flowers in shades of pink, white, and purple.",
    "prince of wales feathers": "Prince of Wales feathers, or Amaranthus caudatus, are ornamental plants with long, drooping flower spikes. They are often used in flower arrangements and add texture to garden beds.",
    "moon orchid": "The moon orchid, or Phalaenopsis amabilis, is a species of orchid native to Southeast Asia. It has large, white flowers with a faint moon-like pattern on its petals.",
    "globe-flower": "The globe-flower, or Trollius europaeus, is a herbaceous perennial plant native to Europe and Asia. It has bright yellow, globe-shaped flowers that bloom in late spring and early summer.",
    "grape hyacinth": "The grape hyacinth, or Muscari, is a small bulbous plant native to Eurasia. It produces clusters of blue, purple, or white flowers that resemble bunches of grapes.",
    "corn poppy": "The corn poppy, or Papaver rhoeas, is a wildflower native to Europe. It has bright red flowers with black spots at the base of each petal.",
    "toad lily": "The toad lily, or Tricyrtis hirta, is a shade-loving perennial native to Japan. It has unique, orchid-like flowers with spotted petals and blooms in late summer and early fall.",
    "siam tulip": "The Siam tulip, or Curcuma alismatifolia, is a tropical plant native to Thailand. It has large, showy flowers in shades of pink, purple, or white.",
    "red ginger": "Red ginger, or Alpinia purpurata, is a tropical plant native to Malaysia. It has striking red inflorescences that resemble torches and is commonly used in Hawaiian leis.",
    "spring crocus": "The spring crocus, or Crocus vernus, is a bulbous perennial native to Europe and Asia. It blooms in early spring, producing colorful flowers in shades of purple, yellow, and white.",
    "alpine sea holly": "Alpine sea holly, or Eryngium alpinum, is a perennial plant native to mountainous regions of Europe. It has spiky blue flowers surrounded by spiny, silvery bracts.",
    "garden phlox": "Garden phlox, or Phlox paniculata, is a popular perennial plant native to North America. It produces large clusters of fragrant flowers in shades of pink, purple, white, and red.",
    "globe thistle": "The globe thistle, or Echinops ritro, is a herbaceous perennial native to Europe and Asia. It has spherical, steel-blue flower heads that attract bees and butterflies.",
    "tiger lily": "The tiger lily, or Lilium lancifolium, is a species of lily native to China and Japan. It has large, orange flowers with dark spots and is commonly grown in gardens for its striking appearance.",
    "ball moss": "Ball moss, or Tillandsia recurvata, is an epiphytic plant native to the southeastern United States, Mexico, and Central America. It has small, gray-green leaves and produces tiny purple flowers.",
    "love in the mist": "Love in the mist, or Nigella damascena, is an annual flowering plant native to southern Europe, North Africa, and southwest Asia. It has delicate, blue or white flowers surrounded by feathery foliage.",
    "monkshood": "Monkshood, or Aconitum napellus, is a perennial plant native to mountainous regions of Europe and Asia. It has tall spikes of hooded, purple flowers and is highly toxic if ingested.",
    "blackberry lily": "The blackberry lily, or Iris domestica, is a species of iris native to China and Japan. It has orange flowers with red or black spots and produces seed pods that resemble blackberries.",
    "spear thistle": "The spear thistle, or Cirsium vulgare, is a biennial or perennial plant native to Europe and western Asia. It has spiny leaves and produces purple, globe-shaped flower heads that attract pollinators.",
    "balloon flower": "Balloon flower, or Platycodon grandiflorus, is a perennial plant native to East Asia. It has balloon-like buds that open to reveal star-shaped flowers in shades of blue, pink, or white.",
    "blanket flower": "The blanket flower, or Gaillardia pulchella, is a perennial plant native to North and South America. It has daisy-like flowers with red or yellow petals and is commonly grown in gardens for its bright colors.",
    "king protea": "The king protea, or Protea cynaroides, is a species of protea native to South Africa. It has large, showy flower heads with long, pink or red bracts that resemble a crown.",
    "oxeye daisy": "The oxeye daisy, or Leucanthemum vulgare, is a perennial plant native to Europe and Asia. It has white, daisy-like flowers with yellow centers and is commonly found in meadows and pastures.",
    "yellow iris": "Yellow iris, or Iris pseudacorus, is a species of iris native to Europe, western Asia, and northern Africa. It has bright yellow flowers with distinctive veining and is commonly found in wetlands and along riverbanks.",
    "cautleya spicata": "Cautleya spicata, or the Himalayan ginger, is a species of flowering plant native to the Himalayas and adjacent regions of Asia. It has yellow or orange flowers and is commonly grown in gardens for its ornamental value.",
    "carnation": "The carnation, or Dianthus caryophyllus, is a herbaceous perennial native to the Mediterranean region. It has fragrant flowers in shades of pink, red, white, and yellow and is commonly used in floral arrangements.",
    "silverbush": "Silverbush, or Convolvulus cneorum, is a low-growing shrub native to the Mediterranean region. It has silvery-gray foliage and produces white, trumpet-shaped flowers that bloom in spring and summer.",
    "bearded iris": "Bearded iris, or Iris germanica, is a species of iris native to Europe and Asia. It has large, showy flowers with frilly petals and a fuzzy 'beard' on the lower petals.",
    "black-eyed susan": "Black-eyed Susan, or Rudbeckia hirta, is a species of coneflower native to North America. It has bright yellow or orange flowers with dark brown centers and is commonly found in gardens and wildflower meadows.",
    "windflower": "Windflower, or Anemone blanda, is a species of flowering plant native to southeastern Europe and Turkey. It has delicate, daisy-like flowers in shades of pink, purple, and white that bloom in early spring.",
    "japanese anemone": "Japanese anemone, or Anemone hupehensis, is a species of flowering plant native to China, Japan, and Korea. It has large, showy flowers in shades of pink or white and blooms in late summer and early fall.",
    "giant white arum lily": "The giant white arum lily, or Zantedeschia aethiopica, is a species of arum lily native to southern Africa. It has large, white flowers with a yellow spadix and is commonly grown as an ornamental plant.",
    "great masterwort": "Great masterwort, or Astrantia major, is a perennial plant native to central and eastern Europe. It has clusters of tiny, star-shaped flowers surrounded by papery bracts and is commonly grown in cottage gardens.",
    "sweet pea": "The sweet pea, or Lathyrus odoratus, is a climbing plant native to Sicily and southern Italy. It has fragrant flowers in shades of pink, purple, blue, and white and is commonly grown for its ornamental value and sweet scent.",

    "tree mallow": "Tree mallow, or Lavatera maritima, is a species of mallow native to the Mediterranean region. It is a woody shrub with pink or white flowers that blooms from summer to fall.",
    "trumpet creeper": "Trumpet creeper, or Campsis radicans, is a species of flowering plant native to the southeastern United States. It has clusters of trumpet-shaped flowers in shades of orange or red and is commonly grown as a climbing vine.",
    "daffodil": "The daffodil, or Narcissus, is a genus of bulbous plants native to Europe and North Africa. They have trumpet-shaped flowers in shades of yellow, white, and orange and are one of the first flowers to bloom in spring.",
    "pincushion flower": "Pincushion flower, or Scabiosa, is a genus of flowering plants native to Europe and Asia. They have globular flower heads in shades of blue, pink, purple, and white and are commonly grown in gardens for their long-lasting blooms.",
    "hard-leaved pocket orchid": "Hard-leaved pocket orchid, or Phalaenopsis equestris, is a species of orchid native to Taiwan. It has small, white flowers with purple or pink markings and is commonly grown as a houseplant.",
    "sunflower": "The sunflower, or Helianthus annuus, is an annual plant native to North and Central America. It has large, yellow flower heads with brown centers and is commonly grown for its edible seeds and oil.",
    "osteospermum": "Osteospermum, or African daisy, is a genus of flowering plants native to southern Africa. They have daisy-like flowers in shades of purple, pink, white, and yellow and are commonly grown as ornamental plants.",
    "tree poppy": "Tree poppy, or Dendromecon rigida, is a species of shrub native to California and northern Mexico. It has bright yellow flowers and is commonly grown in gardens for its ornamental value.",
    "desert-rose": "Desert-rose, or Adenium obesum, is a species of succulent plant native to the Sahel regions of Africa and the Arabian Peninsula. It has pink or white flowers with dark red markings and is commonly grown as a houseplant.",
    "bromelia": "Bromelia is a genus of flowering plants native to tropical regions of the Americas. They have colorful, showy flowers and are commonly grown as ornamental plants in gardens and landscapes.",
    "magnolia": "The magnolia is a genus of flowering plants native to Asia and the Americas. They have large, fragrant flowers in shades of white, pink, and purple and are commonly grown as ornamental trees and shrubs.",
    "english marigold": "The English marigold, or Calendula officinalis, is a species of marigold native to southern Europe. It has bright orange or yellow flowers and is commonly grown in gardens for its medicinal properties and ornamental value.",
    "bee balm": "Bee balm, or Monarda, is a genus of flowering plants native to North America. They have fragrant flowers in shades of pink, purple, red, and white and are commonly grown in gardens for their attractiveness to pollinators.",
    "stemless gentian": "Stemless gentian, or Gentiana acaulis, is a species of flowering plant native to central and southern Europe. It has deep blue, trumpet-shaped flowers and is commonly grown in rock gardens and alpine meadows.",
    "mallow": "Mallow, or Malva, is a genus of flowering plants native to Europe, Asia, and North Africa. They have pink, purple, or white flowers and are commonly grown in gardens for their attractive blooms and edible leaves.",
    "gaura": "Gaura is a genus of flowering plants native to North America. They have delicate, star-shaped flowers in shades of pink and white and are commonly grown in gardens for their long-lasting blooms and drought tolerance.",
    "lenten rose": "Lenten rose, or Helleborus orientalis, is a species of hellebore native to Greece and Turkey. It has nodding flowers in shades of white, pink, purple, and green and blooms in late winter and early spring.",
    "marigold": "Marigold is a genus of annual and perennial flowering plants native to the Americas. They have bright orange, yellow, or red flowers and are commonly grown in gardens as ornamental plants and companion crops.",
    "orange dahlia": "Orange dahlia, or Dahlia, is a genus of flowering plants native to Mexico and Central America. They have large, showy flowers in a wide range of colors, including orange, red, pink, purple, and white.",
    "buttercup": "Buttercup is a genus of flowering plants native to Europe, Asia, and North America. They have bright yellow, cup-shaped flowers and are commonly found in meadows and grasslands.",
    "pelargonium": "Pelargonium, or geranium, is a genus of flowering plants native to southern Africa. They have brightly colored flowers in shades of pink, red, purple, and white and are commonly grown as ornamental plants in gardens and containers.",
    "ruby-lipped cattleya": "Ruby-lipped cattleya, or Cattleya labiata, is a species of orchid native to Brazil. It has large, showy flowers in shades of pink, purple, and white and is commonly grown as a houseplant.",
    "hippeastrum": "Hippeastrum, or amaryllis, is a genus of flowering plants native to South America and the Caribbean. They have large, trumpet-shaped flowers in shades of red, pink, white, and orange and are commonly grown as houseplants and ornamentals.",
    "artichoke": "Artichoke, or Cynara cardunculus var. scolymus, is a species of thistle native to the Mediterranean region. It is cultivated for its edible flower buds and is commonly used in culinary dishes.",
    "gazania": "Gazania is a genus of flowering plants native to southern Africa. They have daisy-like flowers in shades of yellow, orange, pink, and white and are commonly grown in gardens as ornamental plants.",
    "canna lily": "Canna lily, or Canna, is a genus of flowering plants native to the New World tropics. They have large, paddle-shaped leaves and showy flowers in shades of red, orange, yellow, and pink.",
    "peruvian lily": "Peruvian lily, or Alstroemeria, is a genus of flowering plants native to South America. They have colorful, trumpet-shaped flowers and are commonly grown as ornamental plants in gardens and flower arrangements.",
    "mexican petunia": "Mexican petunia, or Ruellia simplex, is a species of flowering plant native to Mexico and the southeastern United States. It has purple or pink flowers and is commonly grown as a ground cover or border plant.",
    "bird of paradise": "Bird of paradise, or Strelitzia reginae, is a species of flowering plant native to South Africa. It has distinctive, bird-like flowers with blue and orange petals and is commonly grown as an ornamental plant in gardens and landscapes.",
    "sweet william": "Sweet William, or Dianthus barbatus, is a species of dianthus native to Europe and Asia. It has fragrant flowers in shades of pink, red, white, and purple and is commonly grown in gardens for its ornamental value.",
    "purple coneflower": "Purple coneflower, or Echinacea purpurea, is a species of coneflower native to eastern North America. It has purple or pink flowers with distinctive cone-shaped centers and is commonly used in herbal medicine and as an ornamental plant.",
    "wild pansy": "Wild pansy, or Viola tricolor, is a species of violet native to Europe and western Asia. It has small, colorful flowers in shades of purple, yellow, and white and is commonly found in meadows and grasslands.",
    "columbine": "Columbine, or Aquilegia, is a genus of perennial plants native to North America, Europe, and Asia. They have distinctive, bell-shaped flowers with spurred petals in shades of blue, purple, pink, red, yellow, and white.",
    "colt's foot": "Colt's foot, or Tussilago farfara, is a species of flowering plant native to Europe and Asia. It has yellow, daisy-like flowers that bloom in early spring and is commonly used in herbal medicine.",
    "snapdragon": "Snapdragon, or Antirrhinum majus, is a species of flowering plant native to the Mediterranean region. It has tall spikes of tubular flowers in shades of pink, red, orange, yellow, and white and is commonly grown in gardens for its colorful blooms.",
    "camellia": "Camellia is a genus of flowering plants native to eastern and southern Asia. They have large, showy flowers in shades of pink, red, white, and yellow and are commonly grown as ornamental shrubs in gardens and landscapes.",
    "fritillary": "Fritillary, or Fritillaria meleagris, is a species of flowering plant native to Europe. It has bell-shaped flowers with distinctive checkerboard patterns in shades of purple, pink, and white and is commonly found in meadows and grasslands.",
    "common dandelion": "The common dandelion, or Taraxacum officinale, is a species of flowering plant native to Europe, Asia, and North America. It has bright yellow flowers and is commonly found in lawns, fields, and other disturbed habitats.",
    "poinsettia": "Poinsettia, or Euphorbia pulcherrima, is a species of flowering plant native to Mexico and Central America. It has bright red bracts surrounding small, yellow flowers and is commonly grown as a decorative houseplant during the holiday season.",
    "primula": "Primula is a genus of flowering plants native to temperate regions of the Northern Hemisphere. They have colorful, cup-shaped flowers in shades of pink, purple, yellow, and white and are commonly grown in gardens and containers.",
    "azalea": "Azalea is a genus of flowering shrubs native to Asia, Europe, and North America. They have showy flowers in shades of pink, red, purple, white, and orange and are commonly grown in gardens for their ornamental value.",
    "californian poppy": "Californian poppy, or Eschscholzia californica, is a species of flowering plant native to California and the southwestern United States. It has bright orange or yellow flowers and is commonly grown as a garden ornamental.",
    "anthurium": "Anthurium is a genus of flowering plants native to tropical regions of the Americas. They have large, heart-shaped leaves and colorful, waxy flowers in shades of red, pink, white, and green and are commonly grown as houseplants.",
    "morning glory": "Morning glory, or Ipomoea, is a genus of flowering plants native to tropical and subtropical regions of the Americas. They have colorful, trumpet-shaped flowers in shades of blue, purple, pink, and white and are commonly grown as ornamental vines.",
    "cape flower": "Cape flower, or Berkheya purpurea, is a species of flowering plant native to southern Africa. It has large, purple flower heads surrounded by spiky, silver-gray leaves and is commonly grown in gardens as an ornamental plant.",
    "bishop of llandaff": "Bishop of Llandaff is a variety of dahlia named after the Bishop of Llandaff, a town in Wales. It has dark red flowers with contrasting dark foliage and is commonly grown in gardens for its striking appearance.",
    "pink-yellow dahlia": "Pink-yellow dahlia, or Dahlia, is a genus of flowering plants native to Mexico and Central America. They have large, showy flowers in a wide range of colors, including pink, yellow, orange, red, purple, and white.",
    "clematis": "Clematis is a genus of flowering vines and shrubs native to temperate regions of the Northern Hemisphere. They have showy flowers in shades of purple, pink, blue, and white and are commonly grown on trellises, fences, and arbors.",
    "geranium": "Geranium is a genus of flowering plants native to temperate regions of the Northern Hemisphere. They have colorful, five-petaled flowers in shades of pink, purple, blue, and white and are commonly grown in gardens as ornamental plants.",
    "thorn apple": "Thorn apple, or Datura stramonium, is a species of flowering plant native to the Americas. It has large, white, trumpet-shaped flowers and spiny seed pods and is highly toxic if ingested.",
    "barbeton daisy": "Barbeton daisy, or Gerbera jamesonii, is a species of flowering plant native to South Africa. It has large, colorful flowers in shades of pink, red, orange, yellow, and white and is commonly grown in gardens for its ornamental value.",
    "bougainvillea": "Bougainvillea is a genus of flowering plants native to South America. They have brightly colored, papery bracts in shades of pink, purple, red, orange, and white and are commonly grown as ornamental vines and shrubs.",
    "sword lily": "Sword lily, or Gladiolus, is a genus of flowering plants native to Africa, Asia, and Europe. They have tall spikes of colorful flowers in shades of red, pink, orange, yellow, and white and are commonly grown in gardens for their ornamental value.",
    "hibiscus": "Hibiscus is a genus of flowering plants native to tropical and subtropical regions around the world. They have large, showy flowers in shades of pink, red, orange, yellow, and white and are commonly grown in gardens and landscapes.",
    "lotus": "Lotus, or Nelumbo nucifera, is a species of aquatic plant native to Asia. It has large, floating leaves and fragrant flowers in shades of pink, white, and yellow and is considered a sacred symbol in many cultures.",
    "cyclamen": "Cyclamen is a genus of flowering plants native to Europe and the Mediterranean region. They have colorful, upswept flowers in shades of pink, purple, red, and white and are commonly grown as houseplants and in gardens.",
    "foxglove": "Foxglove, or Digitalis purpurea, is a species of flowering plant native to Europe. It has tall spikes of tubular flowers in shades of pink, purple, white, and yellow and is highly toxic if ingested.",
    "frangipani": "Frangipani, or Plumeria, is a genus of flowering plants native to tropical regions of the Americas and the Pacific Islands. They have fragrant flowers in shades of white, pink, yellow, and red and are commonly grown as ornamental trees and shrubs.",
    "rose": "Rose is a genus of flowering plants native to Asia, Europe, and North America. They have fragrant flowers in a wide range of colors, including red, pink, yellow, white, and orange, and are commonly grown in gardens and landscapes.",
    "watercress": "Watercress, or Nasturtium officinale, is a species of aquatic plant native to Europe and Asia. It has small, white flowers and peppery, edible leaves and is commonly grown as a leafy green vegetable.",
    "water lily": "Water lily, or Nymphaea, is a genus of aquatic plants native to temperate and tropical regions around the world. They have large, floating leaves and showy flowers in shades of pink, white, yellow, and red and are commonly grown in ponds and water gardens.",
    "wallflower": "Wallflower, or Erysimum cheiri, is a species of flowering plant native to southern Europe. It has clusters of fragrant flowers in shades of yellow, orange, pink, and purple and is commonly grown in gardens for its ornamental value.",
    "passion flower": "Passion flower, or Passiflora, is a genus of flowering plants native to the Americas. They have unique, intricate flowers with colorful, frilly petals and are commonly grown as ornamental vines and shrubs.",
    "petunia": "Petunia is a genus of flowering plants native to South America. They have trumpet-shaped flowers in shades of pink, purple, red, white, and yellow and are commonly grown in gardens and containers as ornamental plants.",
}


def predict_label_and_description(image, model, flower_labels, flower_descriptions):
    # Resize and preprocess the image
    img_array = tf.image.resize(image, [224, 224]) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the label using the provided model
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_label = flower_labels.get(str(predicted_class + 1), "Unknown")

    # Get the description based on the predicted label
    description = flower_descriptions.get(predicted_label, "Description not available")

    return predicted_label, description


def main():
    st.title("Flower Classification")
    st.write("Upload a picture of a flower and let me predict its label and provide a description.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Predict label and description
        label, description = predict_label_and_description(image, model, flower_labels, flower_descriptions)

        # Display prediction in a more visually appealing way
        st.write(f"Label: {label}")
        st.write(f"Description: {description}")

main()