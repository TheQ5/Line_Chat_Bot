from aip import AipSpeech
import json
import time
import wave
from pydub import AudioSegment
# import opencc

#這是轉檔的部分
def convert_aac_to_wav(aacPath):
	    # convert aac to wav
	    aac = AudioSegment.from_file(aacPath)
	    aac.export(aacPath.split(".")[-2].replace("/","") + ".wav", format="wav")

	    # check data
	    voice = wave.open(aacPath.split(".")[-2].replace("/","") + ".wav", "rb")
	    channel, _, framerate, _, _, _ = voice.getparams()
	    print(f"###開始轉檔###\n聲道數:{channel} 音頻:{framerate}\n###轉檔成功###")



#語音轉文字(內含轉檔)
def voice_to_txt(aacPath):

    """ 你的 APPID AK SK """
    APP_ID = '17605237'
    API_KEY = '5yp7MfZ6plHeSXTxVQlpZznF'
    SECRET_KEY = 'DxnTt3LnpumGnpQdYjZWWu34NmDnyOQG'

    begin = time.time()
    client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)


    # 读取文件
    def get_file_content(filepath):
        with open(filepath, 'rb') as fp:
            return fp.read()


    # 识别本地文件
    convert_aac_to_wav(aacPath)
    #把語音轉成文字
    print("語音轉文字中")

    result = client.asr(get_file_content(aacPath), 'wav', 16000, {
        'dev_pid': 1536,
    })
    
    print(result)
    print("Request time cost %f" % (time.time() - begin))
    if result['err_no'] == 0:
        ofile = "result_%s.txt" % (aacPath)
        with open(ofile, "w", encoding="utf-8") as of:
            json.dump(result, of, ensure_ascii=False)
    else:
        print(result['err_msg'], result['err_no'])
    print(result["result"])
    return result["result"][0]




# aacPath = "audios10848253987926.wav"
# voice_to_txt(aacPath)