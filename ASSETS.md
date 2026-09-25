# Fuentes de las imágenes de ejemplo

Las 17 fotografías de `samples/` proceden de Wikimedia Commons. El script
[`preparar_samples.py`](preparar_samples.py) comprueba por API que cada archivo
sigue publicado como [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)
antes de descargarlo. Las copias locales están redimensionadas a un máximo
de 960 píxeles y convertidas a JPEG para el laboratorio. Se acredita a los
autores aunque CC0 no exija atribución.

| Archivo local | Fuente original | Autor indicado en Commons | Licencia |
| --- | --- | --- | --- |
| `samples/perro.jpg` | [Dog in Chacachacare.jpg](https://commons.wikimedia.org/wiki/File:Dog_in_Chacachacare.jpg) | Wilfredor | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| `samples/gato.jpg` | [Tabby cat with blue eyes-3336579.jpg](https://commons.wikimedia.org/wiki/File:Tabby_cat_with_blue_eyes-3336579.jpg) | AdinaVoicu | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| `samples/leon.jpg` | [Lion Ngorongoro 2.jpg](https://commons.wikimedia.org/wiki/File:Lion_Ngorongoro_2.jpg) | Naturedata | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| `samples/panda.jpg` | [Giant panda in Edinburgh Zoo 01.jpg](https://commons.wikimedia.org/wiki/File:Giant_panda_in_Edinburgh_Zoo_01.jpg) | Adam Harangozó | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| `samples/pizza.jpg` | [Tomato onion pizza.jpg](https://commons.wikimedia.org/wiki/File:Tomato_onion_pizza.jpg) | Swathi sri srinivasa raghavan | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| `samples/banano.jpg` | [Banana for human nutrition.jpg](https://commons.wikimedia.org/wiki/File:Banana_for_human_nutrition.jpg) | Synkubito | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| `samples/cafe.jpg` | [Cup of coffee in Café Butter - Prenzlauer Berg, Berlin.jpg](https://commons.wikimedia.org/wiki/File:Cup_of_coffee_in_Caf%C3%A9_Butter_-_Prenzlauer_Berg,_Berlin.jpg) | Edward | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| `samples/manzana.jpg` | [Apples fruit.jpg](https://commons.wikimedia.org/wiki/File:Apples_fruit.jpg) | safaritravelplus | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| `samples/coliseo.jpg` | [Colosseum of Rome, Italy.jpg](https://commons.wikimedia.org/wiki/File:Colosseum_of_Rome,_Italy.jpg) | Wilfredor | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| `samples/torre-eiffel.jpg` | [Tour Eiffel vue du Champ-de-Mars.jpg](https://commons.wikimedia.org/wiki/File:Tour_Eiffel_vue_du_Champ-de-Mars.jpg) | Jebulon | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| `samples/everest.jpg` | [GoWildImages MtEverest NEP0555.jpg](https://commons.wikimedia.org/wiki/File:GoWildImages_MtEverest_NEP0555.jpg) | Jason Freeman / GoWild Images Australia | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| `samples/bicicleta.jpg` | [Bicycle 02.jpg](https://commons.wikimedia.org/wiki/File:Bicycle_02.jpg) | Shams948 | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| `samples/avion.jpg` | [September 2006, airplane in Reykjavik.jpg](https://commons.wikimedia.org/wiki/File:September_2006,_airplane_in_Reykjavik.jpg) | Syced | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| `samples/guitarra.jpg` | [Guitar of the week 28.jpg](https://commons.wikimedia.org/wiki/File:Guitar_of_the_week_28.jpg) | Jesse Canarte | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| `samples/playa-perro.jpg` | [Dog on a beach.jpg](https://commons.wikimedia.org/wiki/File:Dog_on_a_beach.jpg) | Unbeatable101 | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| `samples/girasol.jpg` | [Sunflowers public domain helianthus annuus IMG 2465.jpg](https://commons.wikimedia.org/wiki/File:Sunflowers_public_domain_helianthus_annuus_IMG_2465.jpg) | S.G.S. | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| `samples/tigre.jpg` | [Tijger (Panthera tigris).jpg](https://commons.wikimedia.org/wiki/File:Tijger_(Panthera_tigris).jpg) | Lucawams | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |

Para regenerar las muestras: `.venv/bin/python preparar_samples.py`.
La licencia y la atribución corresponden a cada página original, no al
nombre del archivo local. Si Commons cambia la licencia o retira alguna
imagen, el script se detiene antes de sustituir las copias locales.
