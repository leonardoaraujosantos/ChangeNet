# ChangeNet
Implementation of the ChangeNet [paper](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11130/Varghese_ChangeNet_A_Deep_Learning_Architecture_for_Visual_Change_Detection_ECCVW_2018_paper.pdf).
<Paste>
  
  #### Block Diagrams
  ![alt text](https://raw.githubusercontent.com/leonardoaraujosantos/ChangeNet/master/docs/imgs/block_diagram1.png)
  ![alt text](https://raw.githubusercontent.com/leonardoaraujosantos/ChangeNet/master/docs/imgs/block_diagram2.png)
  
  #### Some Changes from Original paper
  I did some few changes to squeeze some accuracy performance...
  1. Use of Focal Loss instead of CrossEntropy
  2. Use bigger Deconvolutio Network (Instead of FC-->Upsample)
  
  #### Some Results
  ![alt text](https://raw.githubusercontent.com/leonardoaraujosantos/ChangeNet/master/docs/imgs/res_1.png)
  ![alt text](https://raw.githubusercontent.com/leonardoaraujosantos/ChangeNet/master/docs/imgs/ref_2.png)
  ![alt text](https://raw.githubusercontent.com/leonardoaraujosantos/ChangeNet/master/docs/imgs/ref_3.png)
