# galactic_bubble

Single Shot MultiBox Detectorにおける赤外線リング構造の同定に用いたコードを載せていく
詳しくは、各ディレクトリにあるReadMe.txtで


explain directory(ディレクトリの説明)
<pre>
galactic_bubble
 |
 |
 |----------make_ssd_model（the coord to construct SSD : ssdを構築するためのコード）
 |                |
 |                |-------vgg_ver(the CNN part is VGG16：CNN部分がVGG16にしたSSDのコード)
 |                |
 |                |-------original(the CNN part is original : CNN部分がオリジナル)
 |
 |
 |----------make_tarin_data(学習データを作成するためのnotebook、選定をした時のbubbleのカタログ、選定した結果のcsv、)
 |
 |
 |----------play_with_data（the coord to process data：データ加工のためのコード）
 |
 |
 |----------spitzer-bubble
 |
 |
 |
 |

</pre>
