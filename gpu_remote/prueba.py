import torch, time
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
from embed import *
t0=time.time(); modelo(); print(f"carga {time.time()-t0:.1f}s")
(et, dt) = cronometro(embed_text, ["un perro en la playa", "una pizza"])
(ei, di) = cronometro(embed_image, ["samples/perro.jpg", "samples/pizza.jpg"])
print("texto", et.shape, f"{dt:.2f}s", "imagen", ei.shape, f"{di:.2f}s")
print("sim texto x imagen\n", cos(et, ei).round(3))
print("VRAM GB", round(torch.cuda.max_memory_allocated()/1e9, 2), "device", next(modelo().model.parameters()).device if hasattr(modelo(),"model") else "?")
