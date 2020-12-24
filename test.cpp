#include "pp_ocr.h"
#include <opencv2/opencv.hpp>
#include "dict.h"
int Compute(std::string source)
{
  std::cout << source<<"\t"<<source.length() << "\n";
  
  std::string s = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  int w[] ={7, 3, 1};
  int c = 0;
  for (int i = 0; i < source.length(); i++)
  {
    if (source[i] == '<')
      continue;
   int index= s.find(source[i]);
    c += index * w[i % 3];
  }
  c %= 10;
  return c;
}

int main(int argc, char **argv)
{
  cv::Mat img = cv::imread(argv[1]);
  pp_ocr(img);
  std::string line2string="EA68783387CHN9006164M2708090LKMOMBPK<<<<A964";
  std::string haha= line2string.substr(0, 10) + line2string.substr(13, 7) + line2string.substr(21, 22);
  std::string haha1= line2string.substr(0, 9) ;
  std::string haha2= line2string.substr(13, 6) ;
  std::string haha3= line2string.substr(21, 6);
  std::cout << Compute(haha1) << "\n";
  std::cout << Compute(haha2) << "\n";
  std::cout << Compute(haha3) << "\n";
  std::cout << Compute(haha) << "\n";
  std::vector<std::string> a = get_chinese();
  std::cout << a.size() << "\n";
  std::vector<std::string> b = get_english();
  std::cout << b.size() << "\n";

  return 0;
}