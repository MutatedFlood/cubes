# On 3d Classification

### with `ConvLSTM2D`, `Conv3D`, and `LSTM`

#### For how to use `main.py` to create a model quickly, please read the `parser` `help` section(it's near the top!) of `main.py`

`ConvLSTM2D` is primarily used for video classification, however, due to its input format, we can also use it to classify *3D* data.

This project is a comparison between [PointNet][p-net], and a _very_ simple model, that has just 3 layers of `Conv3D`, or `ConvLSTM2D`. Then a `Dropout` layer, followed by a `Flatten` layer, and at last reaches a `Dense` layer with `Softmax` activation.

The loss I use is `categorical_crossentropy`, and there are _40_ classes to choose from. 

**Here are the modules**

>  [`converter`](converter.py): helper module.  
>
>  *	`load_data`: load points in $$(N, points, 3)$$ format. ==usage==: used with [PointNet][p-net] style classification
>  *	`convert`: load points in $$(N, points, 3)​$$ format, then convert it to $$(N, grid\_size, grid\_size,gird\_size, 1 )​$$  format. ==usage==: used with `3D` classification data.
>  *	[`rotate_point_cloud`](converter_copy.py): rotating pointcloud arbitraily about $$z\ \ axis$$ 
>
>  [`provider`](provider.py): helper module. A direct copy from the original [PointNet GitHub Repo][p-net repo]

The basic idea of this project is to replace PointNet with a better module... And we kinda did it. 

**However, it has some limitations.**

1. All data passed in are supposed to be `3D` data. Hence the **overhead** of convertion.
2. The __size__ of the "_field_" has to be pre-known. Cannot be too small.

**The model is as follows**

![model](model.png)

In which **A**, **B**, **C**, **D** can be different kinds of layers, with the rule being **A**, **B**'s input tenosr shape and output tensor shape are `3D`, $$(N, height, width, depth, channel)$$, **C**'s input tensor shape is `3D`, and **D**'s output tensor shape is limited to $$(N, classes)$$. 

#### In the process I accidentally evaluated on train data. The following table shows the results.

|                             kind                             | accuracy on unrotated data* | accurace on rotated data** | average training time (per epoch) |
| :----------------------------------------------------------: | :-------------------------: | :------------------------: | :-----------------------: |
|                           PointNet                           |             .88             |            .81             |            23s            |
|               A,B,C: `ConvLSTM2D`,  D:`Dense`                |            >.99             |          .85\*\*\*           |            83s            |
| A,B,C: `ConvLSTM2D`, `stateful=True`,  D:`Dense`, `l1=.1, l2=1` |            >.99             |          .85\*\*\*           |            83s            |
|                  A,B,C:`Conv3D`, D: `Dense`                  |            >.99             |          .79 \*\*\*           |            2s             |
|                          D: `LSTM`                           |          very low           |          very low          |             ~             |
|           A:`Conv3d`, B,C:`ConvLSTM2D`, D: `Dense`           |             .99             |           .8\*\*\*           |            34s            |
|           B:`Conv3d`, A,C:`ConvLSTM2D`, D: `Dense`           |             .99             |          .82\*\*\*           |            19s            |
|           C:`Conv3d`, A,B:`ConvLSTM2D`, D: `Dense`           |             .99             |          .85\*\*\*           |            31s            |
|           A,B:`Conv3d`, C:`ConvLSTM2D`, D: `Dense`           |             .99             |          .81\*\*\*           |            7.5s           |
|           A,C:`Conv3d`, B:`ConvLSTM2D`, D: `Dense`           |             .99             |          .81\*\*\*           |            19s            |
|           B,C:`Conv3d`, A:`ConvLSTM2D`, D: `Dense`           |             .99             |          .83\*\*\*          |            11s            |

\* training data and testing data are both unrotated, 

\*\* training data and testing data are both rotated.

\*\*\* depends on how many epochs there are for a rotation, **fluctuates** quite a bit during the cycle.

`PointNet` is implemented as in [PointNet-implementation][p-net code], `keras` version.

The model is implemented in [main.py](main.py). Note that **D** being `LSTM` is not implemented, since I don't want to bloat the code. It's very easy, I assure you.

###### So, from the table above, we can see that 3d models typically performs a lot better on unrotated data. While having a slight edge on rotated data, _some_ of the models train _significantly_ faster than _PointNet_, though.

_PointNet actually gets beaten by one of the methods the paper brushes off saying that it's not good enough (Conv3d, both in accuracy and training time)_

**Through this, we accidentally discovered that [PoinNet][ p-net ] isn't really a good performer, gets only 88% of the training data right.**

## However, the above results are incorrect, as I accidentally evaluated on training data.


#### The following is the actual results.

_The simple model I designed didn't have great performances, therefore I dropped them in favor of `Residual Networks` and `SeparableConv2D` implementations, which perform even **worse**._

**Sugessted reason why they perform very poorly:**

1. PointNet's inputs are messy. Every shuffle messes up backpropagation, `ResNets` are more likely to be affected (due to its artifact) ,while `SeparableConv2D` layers aren't able to learn _separated_ features.
2. PointNet's dataset is too small for these nets to learn effectively (I see no sign of _overfit_, though)
3. **WILL** be improved.

|                             kind                             | accuracy on unrotated data\* | accuracy on rotated data\*\* | average training time (per epoch) |
| :----------------------------------------------------------: | :--------------------------: | :--------------------------: | :-------------------------------: |
|                          `PointNet`                          |             .84              |             .77              |                22s                |
|                     `Residual PointNet`                      |             .04              |             .64              |                23s                |
|                     `Separable PointNet`                     |              .4              |              .1              |                25s                |
|                `Separable Residual PointNet`                 |             .64              |             .41              |                27s                |
|               A,B,C: `ConvLSTM2D`,  D:`Dense`                |              .8              |             .76              |                47s                |
|           A:`Conv3D`,  B,C:`ConvLSTM2D`, D:`Dense`           |             .81              |             .75              |                32s                |
|           B:`Conv3d`, A,C:`ConvLSTM2D`, D: `Dense`           |             .82              |             .75              |                16s                |
|           C:`Conv3d`, A,B:`ConvLSTM2D`, D: `Dense`           |             .83              |             .78              |                31s                |
|                  A,B,C:`Conv3D`, D: `Dense`                  |             .83              |             .75              |                2s                 |
|   A,B,C:`Conv3D`, D: `Dense`, `l1_l2`, increased `Dropout`   |             .83              |             .75              |                2s                 |
|           A,B:`Conv3d`, C:`ConvLSTM2D`, D: `Dense`           |             .83              |             .73              |                6s                 |
|           A,C:`Conv3d`, B:`ConvLSTM2D`, D: `Dense`           |             .83              |             .78              |                19s                |
|           B,C:`Conv3d`, A:`ConvLSTM2D`, D: `Dense`           |             .83              |             .77              |                11s                |
|            A,C:`Conv3d`, B:`MaxPool3D, D: `Dense`            |             .86              |             .74              |                2s                 |
| A,C:`Conv3d`, B:`MaxPool3D`, D: `Dense`, smaller pool_size, another`MaxPool3D` added before `Flatten` |             .84              |             .73              |                2s                 |

`PointNet` is implemented as in [PointNet-implementation][p-net code], `keras` version.

`Residual` and `Separable` PointNet implementation can be found in the above repo, too. 

**Please read [_this_][p-net code] for more information**

Recommended **Markdown** viewer: _Typora_

**Hardware** I use: _GTX 1080TI_ * 1, _Intel Core i7 7700_

**Software** I use: `keras` with `tensorflow` backend. In `python3` of course.

[p-net]: https://arxiv.org/abs/1612.00593	"PointNet paper"
[p-net repo]: https://github.com/charlesq34/pointnet " PointNet implementation on GitHub "
[p-net code]: https://github.com/MutatedFlood/pointnet-implementation.git	"PointNet implementation I wrote"