### <p align="center">Multi-granularity Localization Transformer with Collaborative Understanding  <br /> for Referring Multi-Object Tracking

<br>
<div align="center">
  Jiajun&nbsp;Chen</a> <b>&middot;</b>
  Jiacheng&nbsp;Lin</a> <b>&middot;</b>
  Guojin&nbsp;Zhong</a> <b>&middot;</b>
  Yihu&nbsp;Guo</a> <b>&middot;</b>
  Zhiyong&nbsp;Li</a> 
  <br> <br>
</div>

<br>
<p align="center">Code will be released soon. </p>
<br>

<div align=center><img src="assets/network.jpg" /></div>

### Abstract

Referring Multi-Object Tracking (RMOT) involves localizing and tracking specific objects in video frames by utilizing linguistic prompts as references. To enhance the effectiveness of linguistic prompts when training, we introduce a novel Multi-Granularity Localization Transformer with collaborative understanding, termed MGLT. Unlike previous methods focused on visual-language fusion and post-processing, MGLT reevaluates RMOT by preventing linguistic feature attenuation and multi-object collaborative localization. MGLT comprises two key components: Multi-Granularity Implicit Query Bootstrapping (MGIQB) and Multi-Granularity Track-Prompt Alignment (MGTPA). MGIQB ensures that tracking and linguistic features are preserved in later layers of network propagation by bootstrapping the model to generate text-relevant and temporal-enhanced track queries. Simultaneously, MGTPA enhances the model’s localization ability by understanding the relative positions of different referred objects within a frame. Extensive experiments on RMOT benchmarks demonstrate that MGLT achieves state-of-the-art performance. Notably, it shows significant improvements of 2.73%, 7.95% and 3.18% in HOTA, AssA, and IDF1, respectively.


### License

The code is released under the MIT License. It is a short, permissive software license. Basically, you can do whatever you want as long as you include the original copyright and license notice in any copy of the software/source.

### Update

- 2024.03.25 Init repository.
