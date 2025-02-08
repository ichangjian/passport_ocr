//
// Created by fujiayi on 2020/7/5.
//
#include <math.h>
#include "ocr_ppredictor.h"
#include <algorithm>
#include <paddle_api.h>
#include <string>
#include <fstream>
/**
 * "LITE_POWER_HIGH" convert to paddle::lite_api::LITE_POWER_HIGH
 * @param cpu_mode
 * @return
 */
static paddle::lite_api::PowerMode
str_to_cpu_mode(const std::string &cpu_mode)
{
  static std::map<std::string, paddle::lite_api::PowerMode> cpu_mode_map{
      {"LITE_POWER_HIGH", paddle::lite_api::LITE_POWER_HIGH},
      {"LITE_POWER_LOW", paddle::lite_api::LITE_POWER_HIGH},
      {"LITE_POWER_FULL", paddle::lite_api::LITE_POWER_FULL},
      {"LITE_POWER_NO_BIND", paddle::lite_api::LITE_POWER_NO_BIND},
      {"LITE_POWER_RAND_HIGH", paddle::lite_api::LITE_POWER_RAND_HIGH},
      {"LITE_POWER_RAND_LOW", paddle::lite_api::LITE_POWER_RAND_LOW}};
  std::string upper_key;
  std::transform(cpu_mode.cbegin(), cpu_mode.cend(), upper_key.begin(),
                 ::toupper);
  auto index = cpu_mode_map.find(upper_key);
  if (index == cpu_mode_map.end())
  {
    LOGE("cpu_mode not found %s", upper_key.c_str());
    return paddle::lite_api::LITE_POWER_HIGH;
  }
  else
  {
    return index->second;
  }
}
std::vector<std::string> load_words_index(const std::string &file_path)
{
  std::vector<std::string> words;
  std::ifstream infile;
  infile.open(file_path);
  if (infile.is_open())
  {
    while (!infile.eof())
    {
      std::string word;
      infile >> word;
      words.push_back(word);
    }
  }
  LOGE("load words size %d", words.size());
  return words;
}

cv::Mat get_fimage(cv::Mat src, int maxLength, int step)
{
  int width = src.cols;
  int height = src.rows;
  int maxWH = width > height ? width : height;
  float ratio = 1;
  int newWidth = width;
  int newHeight = height;
  if (maxWH > maxLength)
  {
    ratio = maxLength * 1.0f / maxWH;
    newWidth = (int)floor(ratio * width);
    newHeight = (int)floor(ratio * height);
  }

  newWidth = newWidth - newWidth % step;
  if (newWidth == 0)
  {
    newWidth = step;
  }
  newHeight = newHeight - newHeight % step;
  if (newHeight == 0)
  {
    newHeight = step;
  }
  cv::Mat dst;
  cv::resize(src, dst, cv::Size(newWidth, newHeight));
  return dst;
}

// read buffer from file
static std::string ReadFile(const std::string &filename)
{
  std::ifstream ifile(filename.c_str());
  if (!ifile.is_open())
  {
    // LOG(FATAL) << "Open file: [" << filename << "] failed.";
    LOGE("load file error");
  }
  std::ostringstream buf;
  char ch;
  while (buf && ifile.get(ch))
    buf.put(ch);
  ifile.close();
  return buf.str();
}
bool check_mrz(int index, std::string &ch)
{
  //可以改为map优化时间
  std::string dct = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<";
  std::vector<int> indexs = {25, 92, 24, 93, 631, 630, 932, 28, 26, 1108, 1220, 3587, 4243, 1219, 4380, 1218, 5127, 424, 3586, 2696, 5192, 3505, 3638, 3588, 4740, 1309, 4730, 5126, 1216, 1380, 4496, 5404, 5032, 5199, 1217, 4135, 3489};
  for (size_t i = 0; i < indexs.size(); i++)
  {
    if (index == indexs[i])
    {
      ch = dct[i];
      return true;
    }
  }
  return false;
}

int pp_ocr(const cv::Mat &image)
{

  std::string det_model_path = "/sdcard/orc/ch_ppocr_mobile_v1.1_det_prune_opt.nb";
  // std::string rec_model_path = "/sdcard/orc/en_ppocr_mobile_v1.1_rec_opt.nb";
  std::string rec_model_path = "/sdcard/orc/ch_ppocr_mobile_v1.1_rec_quant_opt.nb";
  std::string cls_model_path = "/sdcard/orc/ch_ppocr_mobile_v1.1_cls_quant_opt.nb";
  // std::string det_model_path = "/sdcard/orc/ch_det_mv3_db_opt.nb";
  // std::string rec_model_path = "/sdcard/orc/ch_rec_mv3_crnn_opt.nb";
  // std::string cls_model_path = "/sdcard/orc/cls_opt_arm.nb";
  std::string words_index_path = "/sdcard/orc/ppocr_keys_v1.txt";
  // std::string words_index_path = "/sdcard/orc/ic15_dict.txt";
  std::vector<std::string> words = load_words_index(words_index_path);
  cv::Mat origin = image.clone();
  if (origin.empty())
  {
    return 0;
  }
  LOGE("image size %d, %d,%d", origin.rows, origin.cols, origin.channels());
  int thread_num = 4;
  // std::string cpu_mode = jstring_to_cpp_string(env, j_cpu_mode);
  ppredictor::OCR_Config conf;
  conf.thread_num = thread_num;
  conf.mode = str_to_cpu_mode("LITE_POWER_HIGH");
  ppredictor::OCR_PPredictor *ppredictor =
      new ppredictor::OCR_PPredictor{conf};
  ppredictor->init_from_file(det_model_path, rec_model_path, cls_model_path);

  LOGI("begin to run native forward");

  cv::Mat fimg;
  int channels = 3;

  fimg = get_fimage(origin, 960, 32);

  float inputMean[] = {0.485f, 0.456f, 0.406f};
  float inputStd[] = {1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f};
  int channelIdx[] = {2, 1, 0};
  float *data = new float[fimg.channels() * fimg.rows * fimg.cols];

  {
    cv::imwrite("/sdcard/fimg.jpg", fimg);
    fimg.convertTo(fimg, CV_32FC3, 1.0 / 255);
    std::cout << "========================\n";
    std::cout << fimg.at<cv::Vec3f>(0, 0) << std::endl;
    LOGE("image size %d, %d", fimg.rows, fimg.cols);

    int width = fimg.cols;
    int height = fimg.rows;
    int channelStride[2] = {width * height, width * height * 2};
    for (size_t y = 0; y < fimg.rows; y++)
    {
      for (size_t x = 0; x < fimg.cols; x++)
      {
        cv::Vec3f rgb = fimg.at<cv::Vec3f>(y, x);
        data[y * width + x] = (rgb[channelIdx[0]] - inputMean[0]) / inputStd[0];
        data[y * width + x + channelStride[0]] = (rgb[channelIdx[1]] - inputMean[1]) / inputStd[1];
        data[y * width + x + channelStride[1]] = (rgb[channelIdx[2]] - inputMean[2]) / inputStd[2];
      }
    }
  }

  std::vector<int64_t> dims_arr = {1, channels, fimg.rows, fimg.cols};
  // fimg -= mn;
  // float *data = (float *)fimg.data;
  int64_t buf_len = channels * fimg.rows * fimg.cols;
  // ppredictor->infer_ocr(dims_arr, data, buf_len, NET_OCR, origin);
  std::vector<ppredictor::OCRPredictResult> results =
      ppredictor->infer_ocr(dims_arr, data, buf_len, NET_OCR, origin);
  LOGI("infer_ocr finished with boxes %ld", results.size());
  // 这里将std::vector<ppredictor::OCRPredictResult> 序列化成
  // float数组，传输到java层再反序列化
  std::vector<float> float_arr;
  for (const ppredictor::OCRPredictResult &r : results)
  {
    float_arr.push_back(r.points.size());
    float_arr.push_back(r.word_index.size());
    float_arr.push_back(r.score);
    std::cout << "=========================\n";
    std::cout << r.points.size() << "\t" << r.word_index.size() << "\n";
    std::string MRZ = "";
    for (const std::vector<int> &point : r.points)
    {
      float_arr.push_back(point.at(0));
      float_arr.push_back(point.at(1));
    }
    int i = 0;
    for (int index : r.word_index)
    {
      // std::cout<<index<<"\t";
      // std::string ch;
      // if (check_mrz(index, ch))
      // {
      //   MRZ += ch;
      // }
      float_arr.push_back(index);
      // std::cout << words[index] << "\n";
      MRZ+=words[index]+"\t";
    }
    std::cout << "\n"
              << MRZ << "\n";
  }
  delete ppredictor;
  return 0;
}
