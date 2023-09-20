[![Anurag's GitHub stats](https://github-readme-stats.vercel.app/api?username=ninpei7114)](https://github.com/anuraghazra/github-readme-stats)

# galactic_bubble

Single Shot MultiBox Detectorにおける赤外線リング構造の同定に用いたコード


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
