#include "dict.h"

std::vector<std::string> get_chinese()
{
    std::vector<std::string> CHINESE;
    /*= {"'", "疗", "绚", "诚", "娇", "溜", "题", "贿", "者", "廖", "更", "纳", "加", "奉", "公", "一", "就", "汴", "计", "与", "路", "房", "原", "妇", "2", "0", "8", "-", "7", "其", ">", ":", "]", ",", "，", "骑", "刈", "全", "消", "昏", "傈", "安", "久", "钟", "嗅", "不", "影", "处", "驽", "蜿", "资", "关", "椤", "地", "瘸", "专", "问", "忖", "票", "嫉", "炎", "韵", "要", "月", "田", "节", "陂", "鄙", "捌", "备", "拳", "伺", "眼", "网", "盎", "大", "傍", "心", "东", "愉", "汇", "蹿", "科", "每", "业", "里", "航", "晏", "字", "平", "录", "先", "1", "3", "彤", "鲶", "产", "稍", "督", "腴", "有", "象", "岳", "注", "绍", "在", "泺", "文", "定", "核", "名", "水", "过", "理", "让", "偷", "率", "等", "这", "发", "”", "为", "含", "肥", "酉", "相", "鄱", "七", "编", "猥", "锛", "日", "镀", "蒂", "掰", "倒", "辆", "栾", "栗", "综", "涩", "州", "雌", "滑", "馀", "了", "机", "块", "司", "宰", "甙", "兴", "矽", "抚", "保", "用", "沧", "秩", "如", "收", "息", "滥", "页", "疑", "埠", "!", "！", "姥", "异", "橹", "钇", "向", "下", "跄", "的", "椴", "沫", "国", "绥", "獠", "报", "开", "民", "蜇", "何", "分", "凇", "长", "讥", "藏", "掏", "施", "羽", "中", "讲", "派", "嘟", "人", "提", "浼", "间", "世", "而", "古", "多", "倪", "唇", "饯", "控", "庚", "首", "赛", "蜓", "味", "断", "制", "觉", "技", "替", "艰", "溢", "潮", "夕", "钺", "外", "摘", "枋", "动", "双", "单", "啮", "户", "枇", "确", "锦", "曜", "杜", "或", "能", "效", "霜", "盒", "然", "侗", "电", "晁", "放", "步", "鹃", "新", "杖", "蜂", "吒", "濂", "瞬", "评", "总", "隍", "对", "独", "合", "也", "是", "府", "青", "天", "诲", "墙", "组", "滴", "级", "邀", "帘", "示", "已", "时", "骸", "仄", "泅", "和", "遨", "店", "雇", "疫", "持", "巍", "踮", "境", "只", "亨", "目", "鉴", "崤", "闲", "体", "泄", "杂", "作", "般", "轰", "化", "解", "迂", "诿", "蛭", "璀", "腾", "告", "版", "服", "省", "师", "小", "规", "程", "线", "海", "办", "引", "二", "桧", "牌", "砺", "洄", "裴", "修", "图", "痫", "胡", "许", "犊", "事", "郛", "基", "柴", "呼", "食", "研", "奶", "律", "蛋", "因", "葆", "察", "戏", "褒", "戒", "再", "李", "骁", "工", "貂", "油", "鹅", "章", "啄", "休", "场", "给", "睡", "纷", "豆", "器", "捎", "说", "敏", "学", "会", "浒", "设", "诊", "格", "廓", "查", "来", "霓", "室", "溆", "￠", "诡", "寥", "焕", "舜", "柒", "狐", "回", "戟", "砾", "厄", "实", "翩", "尿", "五", "入", "径", "惭", "喹", "股", "宇", "篝", "|", ";", "美", "期", "云", "九", "祺", "扮", "靠", "锝", "槌", "系", "企", "酰", "阊", "暂", "蚕", "忻", "豁", "本", "羹", "执", "条", "钦", "H", "獒", "限", "进", "季", "楦", "于", "芘", "玖", "铋", "茯", "未", "答", "粘", "括", "样", "精", "欠", "矢", "甥", "帷", "嵩", "扣", "令", "仔", "风", "皈", "行", "支", "部", "蓉", "刮", "站", "蜡", "救", "钊", "汗", "松", "嫌", "成", "可", ".", "鹤", "院", "从", "交", "政", "怕", "活", "调", "球", "局", "验", "髌", "第", "韫", "谗", "串", "到", "圆", "年", "米", "/", "*", "友", "忿", "检", "区", "看", "自", "敢", "刃", "个", "兹", "弄", "流", "留", "同", "没", "齿", "星", "聆", "轼", "湖", "什", "三", "建", "蛔", "儿", "椋", "汕", "震", "颧", "鲤", "跟", "力", "情", "璺", "铨", "陪", "务", "指", "族", "训", "滦", "鄣", "濮", "扒", "商", "箱", "十", "召", "慷", "辗", "所", "莞", "管", "护", "臭", "横", "硒", "嗓", "接", "侦", "六", "露", "党", "馋", "驾", "剖", "高", "侬", "妪", "幂", "猗", "绺", "骐", "央", "酐", "孝", "筝", "课", "徇", "缰", "门", "男", "西", "项", "句", "谙", "瞒", "秃", "篇", "教", "碲", "罚", "声", "呐", "景", "前", "富", "嘴", "鳌", "稀", "免", "朋", "啬", "睐", "去", "赈", "鱼", "住", "肩", "愕", "速", "旁", "波", "厅", "健", "茼", "厥", "鲟", "谅", "投", "攸", "炔", "数", "方", "击", "呋", "谈", "绩", "别", "愫", "僚", "躬", "鹧", "胪", "炳", "招", "喇", "膨", "泵", "蹦", "毛", "结", "5", "4", "谱", "识", "陕", "粽", "婚", "拟", "构", "且", "搜", "任", "潘", "比", "郢", "妨", "醪", "陀", "桔", "碘", "扎", "选", "哈", "骷", "楷", "亿", "明", "缆", "脯", "监", "睫", "逻", "婵", "共", "赴", "淝", "凡", "惦", "及", "达", "揖", "谩", "澹", "减", "焰", "蛹", "番", "祁", "柏", "员", "禄", "怡", "峤", "龙", "白", "叽", "生", "闯", "起", "细", "装", "谕", "竟", "聚", "钙", "上", "导", "渊", "按", "艾", "辘", "挡", "耒", "盹", "饪", "臀", "记", "邮", "蕙", "受", "各", "医", "搂", "普", "滇", "朗", "茸", "带", "翻", "酚", "(", "光", "堤", "墟", "蔷", "万", "幻", "〓", "瑙", "辈", "昧", "盏", "亘", "蛀", "吉", "铰", "请", "子", "假", "闻", "税", "井", "诩", "哨", "嫂", "好", "面", "琐", "校", "馊", "鬣", "缂", "营", "访", "炖", "占", "农", "缀", "否", "经", "钚", "棵", "趟", "张", "亟", "吏", "茶", "谨", "捻", "论", "迸", "堂", "玉", "信", "吧", "瞠", "乡", "姬", "寺", "咬", "溏", "苄", "皿", "意", "赉", "宝", "尔", "钰", "艺", "特", "唳", "踉", "都", "荣", "倚", "登", "荐", "丧", "奇", "涵", "批", "炭", "近", "符", "傩", "感", "道", "着", "菊", "虹", "仲", "众", "懈", "濯", "颞", "眺", "南", "释", "北", "缝", "标", "既", "茗", "整", "撼", "迤", "贲", "挎", "耱", "拒", "某", "妍", "卫", "哇", "英", "矶", "藩", "治", "他", "元", "领", "膜", "遮", "穗", "蛾", "飞", "荒", "棺", "劫", "么", "市", "火", "温", "拈", "棚", "洼", "转", "果", "奕", "卸", "迪", "伸", "泳", "斗", "邡", "侄", "涨", "屯", "萋", "胭", "氡", "崮", "枞", "惧", "冒", "彩", "斜", "手", "豚", "随", "旭", "淑", "妞", "形", "菌", "吲", "沱", "争", "驯", "歹", "挟", "兆", "柱", "传", "至", "包", "内", "响", "临", "红", "功", "弩", "衡", "寂", "禁", "老", "棍", "耆", "渍", "织", "害", "氵", "渑", "布", "载", "靥", "嗬", "虽", "苹", "咨", "娄", "库", "雉", "榜", "帜", "嘲", "套", "瑚", "亲", "簸", "欧", "边", "6", "腿", "旮", "抛", "吹", "瞳", "得", "镓", "梗", "厨", "继", "漾", "愣", "憨", "士", "策", "窑", "抑", "躯", "襟", "脏", "参", "贸", "言", "干", "绸", "鳄", "穷", "藜", "音", "折", "详", ")", "举", "悍", "甸", "癌", "黎", "谴", "死", "罩", "迁", "寒", "驷", "袖", "媒", "蒋", "掘", "模", "纠", "恣", "观", "祖", "蛆", "碍", "位", "稿", "主", "澧", "跌", "筏", "京", "锏", "帝", "贴", "证", "糠", "才", "黄", "鲸", "略", "炯", "饱", "四", "出", "园", "犀", "牧", "容", "汉", "杆", "浈", "汰", "瑷", "造", "虫", "瘩", "怪", "驴", "济", "应", "花", "沣", "谔", "夙", "旅", "价", "矿", "以", "考", "s", "u", "呦", "晒", "巡", "茅", "准", "肟", "瓴", "詹", "仟", "褂", "译", "桌", "混", "宁", "怦", "郑", "抿", "些", "余", "鄂", "饴", "攒", "珑", "群", "阖", "岔", "琨", "藓", "预", "环", "洮", "岌", "宀", "杲", "瀵", "最", "常", "囡", "周", "踊", "女", "鼓", "袭", "喉", "简", "范", "薯", "遐", "疏", "粱", "黜", "禧", "法", "箔", "斤", "遥", "汝", "奥", "直", "贞", "撑", "置", "绱", "集", "她", "馅", "逗", "钧", "橱", "魉", "[", "恙", "躁", "唤", "9", "旺", "膘", "待", "脾", "惫", "购", "吗", "依", "盲", "度", "瘿", "蠖", "俾", "之", "镗", "拇", "鲵", "厝", "簧", "续", "款", "展", "啃", "表", "剔", "品", "钻", "腭", "损", "清", "锶", "统", "涌", "寸", "滨", "贪", "链", "吠", "冈", "伎", "迥", "咏", "吁", "览", "防", "迅", "失", "汾", "阔", "逵", "绀", "蔑", "列", "川", "凭", "努", "熨", "揪", "利", "俱", "绉", "抢", "鸨", "我", "即", "责", "膦", "易", "毓", "鹊", "刹", "玷", "岿", "空", "嘞", "绊", "排", "术", "估", "锷", "违", "们", "苟", "铜", "播", "肘", "件", "烫", "审", "鲂", "广", "像", "铌", "惰", "铟", "巳", "胍", "鲍", "康", "憧", "色", "恢", "想", "拷", "尤", "疳", "知", "S", "Y", "F", "D", "A", "峄", "裕", "帮", "握", "搔", "氐", "氘", "难", "墒", "沮", "雨", "叁", "缥", "悴", "藐", "湫", "娟", "苑", "稠", "颛", "簇", "后", "阕", "闭", "蕤", "缚", "怎", "佞", "码", "嘤", "蔡", "痊", "舱", "螯", "帕", "赫", "昵", "升", "烬", "岫", "、", "疵", "蜻", "髁", "蕨", "隶", "烛", "械", "丑", "盂", "梁", "强", "鲛", "由", "拘", "揉", "劭", "龟", "撤", "钩", "呕", "孛", "费", "妻", "漂", "求", "阑", "崖", "秤", "甘", "通", "深", "补", "赃", "坎", "床", "啪", "承", "吼", "量", "暇", "钼", "烨", "阂", "擎", "脱", "逮", "称", "P", "神", "属", "矗", "华", "届", "狍", "葑", "汹", "育", "患", "窒", "蛰", "佼", "静", "槎", "运", "鳗", "庆", "逝", "曼", "疱", "克", "代", "官", "此", "麸", "耧", "蚌", "晟", "例", "础", "榛", "副", "测", "唰", "缢", "迹", "灬", "霁", "身", "岁", "赭", "扛", "又", "菡", "乜", "雾", "板", "读", "陷", "徉", "贯", "郁", "虑", "变", "钓", "菜", "圾", "现", "琢", "式", "乐", "维", "渔", "浜", "左", "吾", "脑", "钡", "警", "T", "啵", "拴", "偌", "漱", "湿", "硕", "止", "骼", "魄", "积", "燥", "联", "踢", "玛", "则", "窿", "见", "振", "畿", "送", "班", "钽", "您", "赵", "刨", "印", "讨", "踝", "籍", "谡", "舌", "崧", "汽", "蔽", "沪", "酥", "绒", "怖", "财", "帖", "肱", "私", "莎", "勋", "羔", "霸", "励", "哼", "帐", "将", "帅", "渠", "纪", "婴", "娩", "岭", "厘", "滕", "吻", "伤", "坝", "冠", "戊", "隆", "瘁", "介", "涧", "物", "黍", "并", "姗", "奢", "蹑", "掣", "垸", "锴", "命", "箍", "捉", "病", "辖", "琰", "眭", "迩", "艘", "绌", "繁", "寅", "若", "毋", "思", "诉", "类", "诈", "燮", "轲", "酮", "狂", "重", "反", "职", "筱", "县", "委", "磕", "绣", "奖", "晋", "濉", "志", "徽", "肠", "呈", "獐", "坻", "口", "片", "碰", "几", "村", "柿", "劳", "料", "获", "亩", "惕", "晕", "厌", "号", "罢", "池", "正", "鏖", "煨", "家", "棕", "复", "尝", "懋", "蜥", "锅", "岛", "扰", "队", "坠", "瘾", "钬", "@", "卧", "疣", "镇", "譬", "冰", "彷", "频", "黯", "据", "垄", "采", "八", "缪", "瘫", "型", "熹", "砰", "楠", "襁", "箐", "但", "嘶", "绳", "啤", "拍", "盥", "穆", "傲", "洗", "盯", "塘", "怔", "筛", "丿", "台", "恒", "喂", "葛", "永", "￥", "烟", "酒", "桦", "书", "砂", "蚝", "缉", "态", "瀚", "袄", "圳", "轻", "蛛", "超", "榧", "遛", "姒", "奘", "铮", "右", "荽", "望", "偻", "卡", "丶", "氰", "附", "做", "革", "索", "戚", "坨", "桷", "唁", "垅", "榻", "岐", "偎", "坛", "莨", "山", "殊", "微", "骇", "陈", "爨", "推", "嗝", "驹", "澡", "藁", "呤", "卤", "嘻", "糅", "逛", "侵", "郓", "酌", "德", "摇", "※", "鬃", "被", "慨", "殡", "羸", "昌", "泡", "戛", "鞋", "河", "宪", "沿", "玲", "鲨", "翅", "哽", "源", "铅", "语", "照", "邯", "址", "荃", "佬", "顺", "鸳", "町", "霭", "睾", "瓢", "夸", "椁", "晓", "酿", "痈", "咔", "侏", "券", "噎", "湍", "签", "嚷", "离", "午", "尚", "社", "锤", "背", "孟", "使", "浪", "缦", "潍", "鞅", "军", "姹", "驶", "笑", "鳟", "鲁", "》", "孽", "钜", "绿", "洱", "礴", "焯", "椰", "颖", "囔", "乌", "孔", "巴", "互", "性", "椽", "哞", "聘", "昨", "早", "暮", "胶", "炀", "隧", "低", "彗", "昝", "铁", "呓", "氽", "藉", "喔", "癖", "瑗", "姨", "权", "胱", "韦", "堑", "蜜", "酋", "楝", "砝", "毁", "靓", "歙", "锲", "究", "屋", "喳", "骨", "辨", "碑", "武", "鸠", "宫", "辜", "烊", "适", "坡", "殃", "培", "佩", "供", "走", "蜈", "迟", "翼", "况", "姣", "凛", "浔", "吃", "飘", "债", "犟", "金", "促", "苛", "崇", "坂", "莳", "畔", "绂", "兵", "蠕", "斋", "根", "砍", "亢", "欢", "恬", "崔", "剁", "餐", "榫", "快", "扶", "‖", "濒", "缠", "鳜", "当", "彭", "驭", "浦", "篮", "昀", "锆", "秸", "钳", "弋", "娣", "瞑", "夷", "龛", "苫", "拱", "致", "%", "嵊", "障", "隐", "弑", "初", "娓", "抉", "汩", "累", "蓖", "\"", "唬", "助", "苓", "昙", "押", "毙", "破", "城", "郧", "逢", "嚏", "獭", "瞻", "溱", "婿", "赊", "跨", "恼", "璧", "萃", "姻", "貉", "灵", "炉", "密", "氛", "陶", "砸", "谬", "衔", "点", "琛", "沛", "枳", "层", "岱", "诺", "脍", "榈", "埂", "征", "冷", "裁", "打", "蹴", "素", "瘘", "逞", "蛐", "聊", "激", "腱", "萘", "踵", "飒", "蓟", "吆", "取", "咙", "簋", "涓", "矩", "曝", "挺", "揣", "座", "你", "史", "舵", "焱", "尘", "苏", "笈", "脚", "溉", "榨", "诵", "樊", "邓", "焊", "义", "庶", "儋", "蟋", "蒲", "赦", "呷", "杞", "诠", "豪", "还", "试", "颓", "茉", "太", "除", "紫", "逃", "痴", "草", "充", "鳕", "珉", "祗", "墨", "渭", "烩", "蘸", "慕", "璇", "镶", "穴", "嵘", "恶", "骂", "险", "绋", "幕", "碉", "肺", "戳", "刘", "潞", "秣", "纾", "潜", "銮", "洛", "须", "罘", "销", "瘪", "汞", "兮", "屉", "r", "林", "厕", "质", "探", "划", "狸", "殚", "善", "煊", "烹", "〒", "锈", "逯", "宸", "辍", "泱", "柚", "袍", "远", "蹋", "嶙", "绝", "峥", "娥", "缍", "雀", "徵", "认", "镱", "谷", "=", "贩", "勉", "撩", "鄯", "斐", "洋", "非", "祚", "泾", "诒", "饿", "撬", "威", "晷", "搭", "芍", "锥", "笺", "蓦", "候", "琊", "档", "礁", "沼", "卵", "荠", "忑", "朝", "凹", "瑞", "头", "仪", "弧", "孵", "畏", "铆", "突", "衲", "车", "浩", "气", "茂", "悖", "厢", "枕", "酝", "戴", "湾", "邹", "飚", "攘", "锂", "写", "宵", "翁", "岷", "无", "喜", "丈", "挑", "嗟", "绛", "殉", "议", "槽", "具", "醇", "淞", "笃", "郴", "阅", "饼", "底", "壕", "砚", "弈", "询", "缕", "庹", "翟", "零", "筷", "暨", "舟", "闺", "甯", "撞", "麂", "茌", "蔼", "很", "珲", "捕", "棠", "角", "阉", "媛", "娲", "诽", "剿", "尉", "爵", "睬", "韩", "诰", "匣", "危", "糍", "镯", "立", "浏", "阳", "少", "盆", "舔", "擘", "匪", "申", "尬", "铣", "旯", "抖", "赘", "瓯", "居", "ˇ", "哮", "游", "锭", "茏", "歌", "坏", "甚", "秒", "舞", "沙", "仗", "劲", "潺", "阿", "燧", "郭", "嗖", "霏", "忠", "材", "奂", "耐", "跺", "砀", "输", "岖", "媳", "氟", "极", "摆", "灿", "今", "扔", "腻", "枝", "奎", "药", "熄", "吨", "话", "q", "额", "慑", "嘌", "协", "喀", "壳", "埭", "视", "著", "於", "愧", "陲", "翌", "峁", "颅", "佛", "腹", "聋", "侯", "咎", "叟", "秀", "颇", "存", "较", "罪", "哄", "岗", "扫", "栏", "钾", "羌", "己", "璨", "枭", "霉", "煌", "涸", "衿", "键", "镝", "益", "岢", "奏", "连", "夯", "睿", "冥", "均", "糖", "狞", "蹊", "稻", "爸", "刿", "胥", "煜", "丽", "肿", "璃", "掸", "跚", "灾", "垂", "樾", "濑", "乎", "莲", "窄", "犹", "撮", "战", "馄", "软", "络", "显", "鸢", "胸", "宾", "妲", "恕", "埔", "蝌", "份", "遇", "巧", "瞟", "粒", "恰", "剥", "桡", "博", "讯", "凯", "堇", "阶", "滤", "卖", "斌", "骚", "彬", "兑", "磺", "樱", "舷", "两", "娱", "福", "仃", "差", "找", "桁", "÷", "净", "把", "阴", "污", "戬", "雷", "碓", "蕲", "楚", "罡", "焖", "抽", "妫", "咒", "仑", "闱", "尽", "邑", "菁", "爱", "贷", "沥", "鞑", "牡", "嗉", "崴", "骤", "塌", "嗦", "订", "拮", "滓", "捡", "锻", "次", "坪", "杩", "臃", "箬", "融", "珂", "鹗", "宗", "枚", "降", "鸬", "妯", "阄", "堰", "盐", "毅", "必", "杨", "崃", "俺", "甬", "状", "莘", "货", "耸", "菱", "腼", "铸", "唏", "痤", "孚", "澳", "懒", "溅", "翘", "疙", "杷", "淼", "缙", "骰", "喊", "悉", "砻", "坷", "艇", "赁", "界", "谤", "纣", "宴", "晃", "茹", "归", "饭", "梢", "铡", "街", "抄", "肼", "鬟", "苯", "颂", "撷", "戈", "炒", "咆", "茭", "瘙", "负", "仰", "客", "琉", "铢", "封", "卑", "珥", "椿", "镧", "窨", "鬲", "寿", "御", "袤", "铃", "萎", "砖", "餮", "脒", "裳", "肪", "孕", "嫣", "馗", "嵇", "恳", "氯", "江", "石", "褶", "冢", "祸", "阻", "狈", "羞", "银", "靳", "透", "咳", "叼", "敷", "芷", "啥", "它", "瓤", "兰", "痘", "懊", "逑", "肌", "往", "捺", "坊", "甩", "呻", "〃", "沦", "忘", "膻", "祟", "菅", "剧", "崆", "智", "坯", "臧", "霍", "墅", "攻", "眯", "倘", "拢", "骠", "铐", "庭", "岙", "瓠", "′", "缺", "泥", "迢", "捶", "?", "？", "郏", "喙", "掷", "沌", "纯", "秘", "种", "听", "绘", "固", "螨", "团", "香", "盗", "妒", "埚", "蓝", "拖", "旱", "荞", "铀", "血", "遏", "汲", "辰", "叩", "拽", "幅", "硬", "惶", "桀", "漠", "措", "泼", "唑", "齐", "肾", "念", "酱", "虚", "屁", "耶", "旗", "砦", "闵", "婉", "馆", "拭", "绅", "韧", "忏", "窝", "醋", "葺", "顾", "辞", "倜", "堆", "辋", "逆", "玟", "贱", "疾", "董", "惘", "倌", "锕", "淘", "嘀", "莽", "俭", "笏", "绑", "鲷", "杈", "择", "蟀", "粥", "嗯", "驰", "逾", "案", "谪", "褓", "胫", "哩", "昕", "颚", "鲢", "绠", "躺", "鹄", "崂", "儒", "俨", "丝", "尕", "泌", "啊", "萸", "彰", "幺", "吟", "骄", "苣", "弦", "脊", "瑰", "〈", "诛", "镁", "析", "闪", "剪", "侧", "哟", "框", "螃", "守", "嬗", "燕", "狭", "铈", "缮", "概", "迳", "痧", "鲲", "俯", "售", "笼", "痣", "扉", "挖", "满", "咋", "援", "邱", "扇", "歪", "便", "玑", "绦", "峡", "蛇", "叨", "〖", "泽", "胃", "斓", "喋", "怂", "坟", "猪", "该", "蚬", "炕", "弥", "赞", "棣", "晔", "娠", "挲", "狡", "创", "疖", "铕", "镭", "稷", "挫", "弭", "啾", "翔", "粉", "履", "苘", "哦", "楼", "秕", "铂", "土", "锣", "瘟", "挣", "栉", "习", "享", "桢", "袅", "磨", "桂", "谦", "延", "坚", "蔚", "噗", "署", "谟", "猬", "钎", "恐", "嬉", "雒", "倦", "衅", "亏", "璩", "睹", "刻", "殿", "王", "算", "雕", "麻", "丘", "柯", "骆", "丸", "塍", "谚", "添", "鲈", "垓", "桎", "蚯", "芥", "予", "飕", "镦", "谌", "窗", "醚", "菀", "亮", "搪", "莺", "蒿", "羁", "足", "J", "真", "轶", "悬", "衷", "靛", "翊", "掩", "哒", "炅", "掐", "冼", "妮", "l", "谐", "稚", "荆", "擒", "犯", "陵", "虏", "浓", "崽", "刍", "陌", "傻", "孜", "千", "靖", "演", "矜", "钕", "煽", "杰", "酗", "渗", "伞", "栋", "俗", "泫", "戍", "罕", "沾", "疽", "灏", "煦", "芬", "磴", "叱", "阱", "榉", "湃", "蜀", "叉", "醒", "彪", "租", "郡", "篷", "屎", "良", "垢", "隗", "弱", "陨", "峪", "砷", "掴", "颁", "胎", "雯", "绵", "贬", "沐", "撵", "隘", "篙", "暖", "曹", "陡", "栓", "填", "臼", "彦", "瓶", "琪", "潼", "哪", "鸡", "摩", "啦", "俟", "锋", "域", "耻", "蔫", "疯", "纹", "撇", "毒", "绶", "痛", "酯", "忍", "爪", "赳", "歆", "嘹", "辕", "烈", "册", "朴", "钱", "吮", "毯", "癜", "娃", "谀", "邵", "厮", "炽", "璞", "邃", "丐", "追", "词", "瓒", "忆", "轧", "芫", "谯", "喷", "弟", "半", "冕", "裙", "掖", "墉", "绮", "寝", "苔", "势", "顷", "褥", "切", "衮", "君", "佳", "嫒", "蚩", "霞", "佚", "洙", "逊", "镖", "暹", "唛", "&", "殒", "顶", "碗", "獗", "轭", "铺", "蛊", "废", "恹", "汨", "崩", "珍", "那", "杵", "曲", "纺", "夏", "薰", "傀", "闳", "淬", "姘", "舀", "拧", "卷", "楂", "恍", "讪", "厩", "寮", "篪", "赓", "乘", "灭", "盅", "鞣", "沟", "慎", "挂", "饺", "鼾", "杳", "树", "缨", "丛", "絮", "娌", "臻", "嗳", "篡", "侩", "述", "衰", "矛", "圈", "蚜", "匕", "筹", "匿", "濞", "晨", "叶", "骋", "郝", "挚", "蚴", "滞", "增", "侍", "描", "瓣", "吖", "嫦", "蟒", "匾", "圣", "赌", "毡", "癞", "恺", "百", "曳", "需", "篓", "肮", "庖", "帏", "卿", "驿", "遗", "蹬", "鬓", "骡", "歉", "芎", "胳", "屐", "禽", "烦", "晌", "寄", "媾", "狄", "翡", "苒", "船", "廉", "终", "痞", "殇", "々", "畦", "饶", "改", "拆", "悻", "萄", "￡", "瓿", "乃", "訾", "桅", "匮", "溧", "拥", "纱", "铍", "骗", "蕃", "龋", "缬", "父", "佐", "疚", "栎", "醍", "掳", "蓄", "x", "惆", "颜", "鲆", "榆", "〔", "猎", "敌", "暴", "谥", "鲫", "贾", "罗", "玻", "缄", "扦", "芪", "癣", "落", "徒", "臾", "恿", "猩", "托", "邴", "肄", "牵", "春", "陛", "耀", "刊", "拓", "蓓", "邳", "堕", "寇", "枉", "淌", "啡", "湄", "兽", "酷", "萼", "碚", "濠", "萤", "夹", "旬", "戮", "梭", "琥", "椭", "昔", "勺", "蜊", "绐", "晚", "孺", "僵", "宣", "摄", "冽", "旨", "萌", "忙", "蚤", "眉", "噼", "蟑", "付", "契", "瓜", "悼", "颡", "壁", "曾", "窕", "颢", "澎", "仿", "俑", "浑", "嵌", "浣", "乍", "碌", "褪", "乱", "蔟", "隙", "玩", "剐", "葫", "箫", "纲", "围", "伐", "决", "伙", "漩", "瑟", "刑", "肓", "镳", "缓", "蹭", "氨", "皓", "典", "畲", "坍", "铑", "檐", "塑", "洞", "倬", "储", "胴", "淳", "戾", "吐", "灼", "惺", "妙", "毕", "珐", "缈", "虱", "盖", "羰", "鸿", "磅", "谓", "髅", "娴", "苴", "唷", "蚣", "霹", "抨", "贤", "唠", "犬", "誓", "逍", "庠", "逼", "麓", "籼", "釉", "呜", "碧", "秧", "氩", "摔", "霄", "穸", "纨", "辟", "妈", "映", "完", "牛", "缴", "嗷", "炊", "恩", "荔", "茆", "掉", "紊", "慌", "莓", "羟", "阙", "萁", "磐", "另", "蕹", "辱", "鳐", "湮", "吡", "吩", "唐", "睦", "垠", "舒", "圜", "冗", "瞿", "溺", "芾", "囱", "匠", "僳", "汐", "菩", "饬", "漓", "黑", "霰", "浸", "濡", "窥", "毂", "蒡", "兢", "驻", "鹉", "芮", "诙", "迫", "雳", "厂", "忐", "臆", "猴", "鸣", "蚪", "栈", "箕", "羡", "渐", "莆", "捍", "眈", "哓", "趴", "蹼", "埕", "嚣", "骛", "宏", "淄", "斑", "噜", "严", "瑛", "垃", "椎", "诱", "压", "庾", "绞", "焘", "廿", "抡", "迄", "棘", "夫", "纬", "锹", "眨", "瞌", "侠", "脐", "竞", "瀑", "孳", "骧", "遁", "姜", "颦", "荪", "滚", "萦", "伪", "逸", "粳", "爬", "锁", "矣", "役", "趣", "洒", "颔", "诏", "逐", "奸", "甭", "惠", "攀", "蹄", "泛", "尼", "拼", "阮", "鹰", "亚", "颈", "惑", "勒", "〉", "际", "肛", "爷", "刚", "钨", "丰", "养", "冶", "鲽", "辉", "蔻", "画", "覆", "皴", "妊", "麦", "返", "醉", "皂", "擀", "〗", "酶", "凑", "粹", "悟", "诀", "硖", "港", "卜", "z", "杀", "涕", "±", "舍", "铠", "抵", "弛", "段", "敝", "镐", "奠", "拂", "轴", "跛", "袱", "e", "t", "沉", "菇", "俎", "薪", "峦", "秭", "蟹", "历", "盟", "菠", "寡", "液", "肢", "喻", "染", "裱", "悱", "抱", "氙", "赤", "捅", "猛", "跑", "氮", "谣", "仁", "尺", "辊", "窍", "烙", "衍", "架", "擦", "倏", "璐", "瑁", "币", "楞", "胖", "夔", "趸", "邛", "惴", "饕", "虔", "蝎", "§", "哉", "贝", "宽", "辫", "炮", "扩", "饲", "籽", "魏", "菟", "锰", "伍", "猝", "末", "琳", "哚", "蛎", "邂", "呀", "姿", "鄞", "却", "歧", "仙", "恸", "椐", "森", "牒", "寤", "袒", "婆", "虢", "雅", "钉", "朵", "贼", "欲", "苞", "寰", "故", "龚", "坭", "嘘", "咫", "礼", "硷", "兀", "睢", "汶", "’", "铲", "烧", "绕", "诃", "浃", "钿", "哺", "柜", "讼", "颊", "璁", "腔", "洽", "咐", "脲", "簌", "筠", "镣", "玮", "鞠", "谁", "兼", "姆", "挥", "梯", "蝴", "谘", "漕", "刷", "躏", "宦", "弼", "b", "垌", "劈", "麟", "莉", "揭", "笙", "渎", "仕", "嗤", "仓", "配", "怏", "抬", "错", "泯", "镊", "孰", "猿", "邪", "仍", "秋", "鼬", "壹", "歇", "吵", "炼", "<", "尧", "射", "柬", "廷", "胧", "霾", "凳", "隋", "肚", "浮", "梦", "祥", "株", "堵", "退", "L", "鹫", "跎", "凶", "毽", "荟", "炫", "栩", "玳", "甜", "沂", "鹿", "顽", "伯", "爹", "赔", "蛴", "徐", "匡", "欣", "狰", "缸", "雹", "蟆", "疤", "默", "沤", "啜", "痂", "衣", "禅", "w", "i", "h", "辽", "葳", "黝", "钗", "停", "沽", "棒", "馨", "颌", "肉", "吴", "硫", "悯", "劾", "娈", "马", "啧", "吊", "悌", "镑", "峭", "帆", "瀣", "涉", "咸", "疸", "滋", "泣", "翦", "拙", "癸", "钥", "蜒", "+", "尾", "庄", "凝", "泉", "婢", "渴", "谊", "乞", "陆", "锉", "糊", "鸦", "淮", "I", "B", "N", "晦", "弗", "乔", "庥", "葡", "尻", "席", "橡", "傣", "渣", "拿", "惩", "麋", "斛", "缃", "矮", "蛏", "岘", "鸽", "姐", "膏", "催", "奔", "镒", "喱", "蠡", "摧", "钯", "胤", "柠", "拐", "璋", "鸥", "卢", "荡", "倾", "^", "_", "珀", "逄", "萧", "塾", "掇", "贮", "笆", "聂", "圃", "冲", "嵬", "M", "滔", "笕", "值", "炙", "偶", "蜱", "搐", "梆", "汪", "蔬", "腑", "鸯", "蹇", "敞", "绯", "仨", "祯", "谆", "梧", "糗", "鑫", "啸", "豺", "囹", "猾", "巢", "柄", "瀛", "筑", "踌", "沭", "暗", "苁", "鱿", "蹉", "脂", "蘖", "牢", "热", "木", "吸", "溃", "宠", "序", "泞", "偿", "拜", "檩", "厚", "朐", "毗", "螳", "吞", "媚", "朽", "担", "蝗", "橘", "畴", "祈", "糟", "盱", "隼", "郜", "惜", "珠", "裨", "铵", "焙", "琚", "唯", "咚", "噪", "骊", "丫", "滢", "勤", "棉", "呸", "咣", "淀", "隔", "蕾", "窈", "饨", "挨", "煅", "短", "匙", "粕", "镜", "赣", "撕", "墩", "酬", "馁", "豌", "颐", "抗", "酣", "氓", "佑", "搁", "哭", "递", "耷", "涡", "桃", "贻", "碣", "截", "瘦", "昭", "镌", "蔓", "氚", "甲", "猕", "蕴", "蓬", "散", "拾", "纛", "狼", "猷", "铎", "埋", "旖", "矾", "讳", "囊", "糜", "迈", "粟", "蚂", "紧", "鲳", "瘢", "栽", "稼", "羊", "锄", "斟", "睁", "桥", "瓮", "蹙", "祉", "醺", "鼻", "昱", "剃", "跳", "篱", "跷", "蒜", "翎", "宅", "晖", "嗑", "壑", "峻", "癫", "屏", "狠", "陋", "袜", "途", "憎", "祀", "莹", "滟", "佶", "溥", "臣", "约", "盛", "峰", "磁", "慵", "婪", "拦", "莅", "朕", "鹦", "粲", "裤", "哎", "疡", "嫖", "琵", "窟", "堪", "谛", "嘉", "儡", "鳝", "斩", "郾", "驸", "酊", "妄", "胜", "贺", "徙", "傅", "噌", "钢", "栅", "庇", "恋", "匝", "巯", "邈", "尸", "锚", "粗", "佟", "蛟", "薹", "纵", "蚊", "郅", "绢", "锐", "苗", "俞", "篆", "淆", "膀", "鲜", "煎", "诶", "秽", "寻", "涮", "刺", "怀", "噶", "巨", "褰", "魅", "灶", "灌", "桉", "藕", "谜", "舸", "薄", "搀", "恽", "借", "牯", "痉", "渥", "愿", "亓", "耘", "杠", "柩", "锔", "蚶", "钣", "珈", "喘", "蹒", "幽", "赐", "稗", "晤", "莱", "泔", "扯", "肯", "菪", "裆", "腩", "豉", "疆", "骜", "腐", "倭", "珏", "唔", "粮", "亡", "润", "慰", "伽", "橄", "玄", "誉", "醐", "胆", "龊", "粼", "塬", "陇", "彼", "削", "嗣", "绾", "芽", "妗", "垭", "瘴", "爽", "薏", "寨", "龈", "泠", "弹", "赢", "漪", "猫", "嘧", "涂", "恤", "圭", "茧", "烽", "屑", "痕", "巾", "赖", "荸", "凰", "腮", "畈", "亵", "蹲", "偃", "苇", "澜", "艮", "换", "骺", "烘", "苕", "梓", "颉", "肇", "哗", "悄", "氤", "涠", "葬", "屠", "鹭", "植", "竺", "佯", "诣", "鲇", "瘀", "鲅", "邦", "移", "滁", "冯", "耕", "癔", "戌", "茬", "沁", "巩", "悠", "湘", "洪", "痹", "锟", "循", "谋", "腕", "鳃", "钠", "捞", "焉", "迎", "碱", "伫", "急", "榷", "奈", "邝", "卯", "辄", "皲", "卟", "醛", "畹", "忧", "稳", "雄", "昼", "缩", "阈", "睑", "扌", "耗", "曦", "涅", "捏", "瞧", "邕", "淖", "漉", "铝", "耦", "禹", "湛", "喽", "莼", "琅", "诸", "苎", "纂", "硅", "始", "嗨", "傥", "燃", "臂", "赅", "嘈", "呆", "贵", "屹", "壮", "肋", "亍", "蚀", "卅", "豹", "腆", "邬", "迭", "浊", "}", "童", "螂", "捐", "圩", "勐", "触", "寞", "汊", "壤", "荫", "膺", "渌", "芳", "懿", "遴", "螈", "泰", "蓼", "蛤", "茜", "舅", "枫", "朔", "膝", "眙", "避", "梅", "判", "鹜", "璜", "牍", "缅", "垫", "藻", "黔", "侥", "惚", "懂", "踩", "腰", "腈", "札", "丞", "唾", "慈", "顿", "摹", "荻", "琬", "~", "斧", "沈", "滂", "胁", "胀", "幄", "莜", "Z", "匀", "鄄", "掌", "绰", "茎", "焚", "赋", "萱", "谑", "汁", "铒", "瞎", "夺", "蜗", "野", "娆", "冀", "弯", "篁", "懵", "灞", "隽", "芡", "脘", "俐", "辩", "芯", "掺", "喏", "膈", "蝈", "觐", "悚", "踹", "蔗", "熠", "鼠", "呵", "抓", "橼", "峨", "畜", "缔", "禾", "崭", "弃", "熊", "摒", "凸", "拗", "穹", "蒙", "抒", "祛", "劝", "闫", "扳", "阵", "醌", "踪", "喵", "侣", "搬", "仅", "荧", "赎", "蝾", "琦", "买", "婧", "瞄", "寓", "皎", "冻", "赝", "箩", "莫", "瞰", "郊", "笫", "姝", "筒", "枪", "遣", "煸", "袋", "舆", "痱", "涛", "母", "〇", "启", "践", "耙", "绲", "盘", "遂", "昊", "搞", "槿", "诬", "纰", "泓", "惨", "檬", "亻", "越", "C", "o", "憩", "熵", "祷", "钒", "暧", "塔", "阗", "胰", "咄", "娶", "魔", "琶", "钞", "邻", "扬", "杉", "殴", "咽", "弓", "〆", "髻", "】", "吭", "揽", "霆", "拄", "殖", "脆", "彻", "岩", "芝", "勃", "辣", "剌", "钝", "嘎", "甄", "佘", "皖", "伦", "授", "徕", "憔", "挪", "皇", "庞", "稔", "芜", "踏", "溴", "兖", "卒", "擢", "饥", "鳞", "煲", "‰", "账", "颗", "叻", "斯", "捧", "鳍", "琮", "讹", "蛙", "纽", "谭", "酸", "兔", "莒", "睇", "伟", "觑", "羲", "嗜", "宜", "褐", "旎", "辛", "卦", "诘", "筋", "鎏", "溪", "挛", "熔", "阜", "晰", "鳅", "丢", "奚", "灸", "呱", "献", "陉", "黛", "鸪", "甾", "萨", "疮", "拯", "洲", "疹", "辑", "叙", "恻", "谒", "允", "柔", "烂", "氏", "逅", "漆", "拎", "惋", "扈", "湟", "纭", "啕", "掬", "擞", "哥", "忽", "涤", "鸵", "靡", "郗", "瓷", "扁", "廊", "怨", "雏", "钮", "敦", "E", "懦", "憋", "汀", "拚", "啉", "腌", "岸", "f", "痼", "瞅", "尊", "咀", "眩", "飙", "忌", "仝", "迦", "熬", "毫", "胯", "篑", "茄", "腺", "凄", "舛", "碴", "锵", "诧", "羯", "後", "漏", "汤", "宓", "仞", "蚁", "壶", "谰", "皑", "铄", "棰", "罔", "辅", "晶", "苦", "牟", "闽", "\\", "烃", "饮", "聿", "丙", "蛳", "朱", "煤", "涔", "鳖", "犁", "罐", "荼", "砒", "淦", "妤", "黏", "戎", "孑", "婕", "瑾", "戢", "钵", "枣", "捋", "砥", "衩", "狙", "桠", "稣", "阎", "肃", "梏", "诫", "孪", "昶", "婊", "衫", "嗔", "侃", "塞", "蜃", "樵", "峒", "貌", "屿", "欺", "缫", "阐", "栖", "诟", "珞", "荭", "吝", "萍", "嗽", "恂", "啻", "蜴", "磬", "峋", "俸", "豫", "谎", "徊", "镍", "韬", "魇", "晴", "U", "囟", "猜", "蛮", "坐", "囿", "伴", "亭", "肝", "佗", "蝠", "妃", "胞", "滩", "榴", "氖", "垩", "苋", "砣", "扪", "馏", "姓", "轩", "厉", "夥", "侈", "禀", "垒", "岑", "赏", "钛", "辐", "痔", "披", "纸", "碳", "“", "坞", "蠓", "挤", "荥", "沅", "悔", "铧", "帼", "蒌", "蝇", "a", "p", "y", "n", "g", "哀", "浆", "瑶", "凿", "桶", "馈", "皮", "奴", "苜", "佤", "伶", "晗", "铱", "炬", "优", "弊", "氢", "恃", "甫", "攥", "端", "锌", "灰", "稹", "炝", "曙", "邋", "亥", "眶", "碾", "拉", "萝", "绔", "捷", "浍", "腋", "姑", "菖", "凌", "涞", "麽", "锢", "桨", "潢", "绎", "镰", "殆", "锑", "渝", "铬", "困", "绽", "觎", "匈", "糙", "暑", "裹", "鸟", "盔", "肽", "迷", "綦", "『", "亳", "佝", "俘", "钴", "觇", "骥", "仆", "疝", "跪", "婶", "郯", "瀹", "唉", "脖", "踞", "针", "晾", "忒", "扼", "瞩", "叛", "椒", "疟", "嗡", "邗", "肆", "跆", "玫", "忡", "捣", "咧", "唆", "艄", "蘑", "潦", "笛", "阚", "沸", "泻", "掊", "菽", "贫", "斥", "髂", "孢", "镂", "赂", "麝", "鸾", "屡", "衬", "苷", "恪", "叠", "希", "粤", "爻", "喝", "茫", "惬", "郸", "绻", "庸", "撅", "碟", "宄", "妹", "膛", "叮", "饵", "崛", "嗲", "椅", "冤", "搅", "咕", "敛", "尹", "垦", "闷", "蝉", "霎", "勰", "败", "蓑", "泸", "肤", "鹌", "幌", "焦", "浠", "鞍", "刁", "舰", "乙", "竿", "裔", "。", "茵", "函", "伊", "兄", "丨", "娜", "匍", "謇", "莪", "宥", "似", "蝽", "翳", "酪", "翠", "粑", "薇", "祢", "骏", "赠", "叫", "Q", "噤", "噻", "竖", "芗", "莠", "潭", "俊", "羿", "耜", "O", "郫", "趁", "嗪", "囚", "蹶", "芒", "洁", "笋", "鹑", "敲", "硝", "啶", "堡", "渲", "揩", "』", "携", "宿", "遒", "颍", "扭", "棱", "割", "萜", "蔸", "葵", "琴", "捂", "饰", "衙", "耿", "掠", "募", "岂", "窖", "涟", "蔺", "瘤", "柞", "瞪", "怜", "匹", "距", "楔", "炜", "哆", "秦", "缎", "幼", "茁", "绪", "痨", "恨", "楸", "娅", "瓦", "桩", "雪", "嬴", "伏", "榔", "妥", "铿", "拌", "眠", "雍", "缇", "‘", "卓", "搓", "哌", "觞", "噩", "屈", "哧", "髓", "咦", "巅", "娑", "侑", "淫", "膳", "祝", "勾", "姊", "莴", "胄", "疃", "薛", "蜷", "胛", "巷", "芙", "芋", "熙", "闰", "勿", "窃", "狱", "剩", "钏", "幢", "陟", "铛", "慧", "靴", "耍", "k", "浙", "浇", "飨", "惟", "绗", "祜", "澈", "啼", "咪", "磷", "摞", "诅", "郦", "抹", "跃", "壬", "吕", "肖", "琏", "颤", "尴", "剡", "抠", "凋", "赚", "泊", "津", "宕", "殷", "倔", "氲", "漫", "邺", "涎", "怠", "$", "垮", "荬", "遵", "俏", "叹", "噢", "饽", "蜘", "孙", "筵", "疼", "鞭", "羧", "牦", "箭", "潴", "c", "眸", "祭", "髯", "啖", "坳", "愁", "芩", "驮", "倡", "巽", "穰", "沃", "胚", "怒", "凤", "槛", "剂", "趵", "嫁", "v", "邢", "灯", "鄢", "桐", "睽", "檗", "锯", "槟", "婷", "嵋", "圻", "诗", "蕈", "颠", "遭", "痢", "芸", "怯", "馥", "竭", "锗", "徜", "恭", "遍", "籁", "剑", "嘱", "苡", "龄", "僧", "桑", "潸", "弘", "澶", "楹", "悲", "讫", "愤", "腥", "悸", "谍", "椹", "呢", "桓", "葭", "攫", "阀", "翰", "躲", "敖", "柑", "郎", "笨", "橇", "呃", "魁", "燎", "脓", "葩", "磋", "垛", "玺", "狮", "沓", "砜", "蕊", "锺", "罹", "蕉", "翱", "虐", "闾", "巫", "旦", "茱", "嬷", "枯", "鹏", "贡", "芹", "汛", "矫", "绁", "拣", "禺", "佃", "讣", "舫", "惯", "乳", "趋", "疲", "挽", "岚", "虾", "衾", "蠹", "蹂", "飓", "氦", "铖", "孩", "稞", "瑜", "壅", "掀", "勘", "妓", "畅", "髋", "W", "庐", "牲", "蓿", "榕", "练", "垣", "唱", "邸", "菲", "昆", "婺", "穿", "绡", "麒", "蚱", "掂", "愚", "泷", "涪", "漳", "妩", "娉", "榄", "讷", "觅", "旧", "藤", "煮", "呛", "柳", "腓", "叭", "庵", "烷", "阡", "罂", "蜕", "擂", "猖", "咿", "媲", "脉", "【", "沏", "貅", "黠", "熏", "哲", "烁", "坦", "酵", "兜", "×", "潇", "撒", "剽", "珩", "圹", "乾", "摸", "樟", "帽", "嗒", "襄", "魂", "轿", "憬", "锡", "〕", "喃", "皆", "咖", "隅", "脸", "残", "泮", "袂", "鹂", "珊", "囤", "捆", "咤", "误", "徨", "闹", "淙", "芊", "淋", "怆", "囗", "拨", "梳", "渤", "R", "G", "绨", "蚓", "婀", "幡", "狩", "麾", "谢", "唢", "裸", "旌", "伉", "纶", "裂", "驳", "砼", "咛", "澄", "樨", "蹈", "宙", "澍", "倍", "貔", "操", "勇", "蟠", "摈", "砧", "虬", "够", "缁", "悦", "藿", "撸", "艹", "摁", "淹", "豇", "虎", "榭", "ˉ", "吱", "d", "°", "喧", "荀", "踱", "侮", "奋", "偕", "饷", "犍", "惮", "坑", "璎", "徘", "宛", "妆", "袈", "倩", "窦", "昂", "荏", "乖", "K", "怅", "撰", "鳙", "牙", "袁", "酞", "X", "痿", "琼", "闸", "雁", "趾", "荚", "虻", "涝", "《", "杏", "韭", "偈", "烤", "绫", "鞘", "卉", "症", "遢", "蓥", "诋", "杭", "荨", "匆", "竣", "簪", "辙", "敕", "虞", "丹", "缭", "咩", "黟", "m", "淤", "瑕", "咂", "铉", "硼", "茨", "嶂", "痒", "畸", "敬", "涿", "粪", "窘", "熟", "叔", "嫔", "盾", "忱", "裘", "憾", "梵", "赡", "珙", "咯", "娘", "庙", "溯", "胺", "葱", "痪", "摊", "荷", "卞", "乒", "髦", "寐", "铭", "坩", "胗", "枷", "爆", "溟", "嚼", "羚", "砬", "轨", "惊", "挠", "罄", "竽", "菏", "氧", "浅", "楣", "盼", "枢", "炸", "阆", "杯", "谏", "噬", "淇", "渺", "俪", "秆", "墓", "泪", "跻", "砌", "痰", "垡", "渡", "耽", "釜", "讶", "鳎", "煞", "呗", "韶", "舶", "绷", "鹳", "缜", "旷", "铊", "皱", "龌", "檀", "霖", "奄", "槐", "艳", "蝶", "旋", "哝", "赶", "骞", "蚧", "腊", "盈", "丁", "`", "蜚", "矸", "蝙", "睨", "嚓", "僻", "鬼", "醴", "夜", "彝", "磊", "笔", "拔", "栀", "糕", "厦", "邰", "纫", "逭", "纤", "眦", "膊", "馍", "躇", "烯", "蘼", "冬", "诤", "暄", "骶", "哑", "瘠", "」", "臊", "丕", "愈", "咱", "螺", "擅", "跋", "搏", "硪", "谄", "笠", "淡", "嘿", "骅", "谧", "鼎", "皋", "姚", "歼", "蠢", "驼", "耳", "胬", "挝", "涯", "狗", "蒽", "孓", "犷", "凉", "芦", "箴", "铤", "孤", "嘛", "坤", "V", "茴", "朦", "挞", "尖", "橙", "诞", "搴", "碇", "洵", "浚", "帚", "蜍", "漯", "柘", "嚎", "讽", "芭", "荤", "咻", "祠", "秉", "跖", "埃", "吓", "糯", "眷", "馒", "惹", "娼", "鲑", "嫩", "讴", "轮", "瞥", "靶", "褚", "乏", "缤", "宋", "帧", "删", "驱", "碎", "扑", "俩", "俄", "偏", "涣", "竹", "噱", "皙", "佰", "渚", "唧", "斡", "#", "镉", "刀", "崎", "筐", "佣", "夭", "贰", "肴", "峙", "哔", "艿", "匐", "牺", "镛", "缘", "仡", "嫡", "劣", "枸", "堀", "梨", "簿", "鸭", "蒸", "亦", "稽", "浴", "{", "衢", "束", "槲", "j", "阁", "揍", "疥", "棋", "潋", "聪", "窜", "乓", "睛", "插", "冉", "阪", "苍", "搽", "「", "蟾", "螟", "幸", "仇", "樽", "撂", "慢", "跤", "幔", "俚", "淅", "覃", "觊", "溶", "妖", "帛", "侨", "曰", "妾", "泗", "·", "：", "瀘", "風", "Ë", "（", "）", "∶", "紅", "紗", "瑭", "雲", "頭", "鶏", "財", "許", "•", "¥", "樂", "焗", "麗", "—", "；", "滙", "東", "榮", "繪", "興", "…", "門", "業", "π", "楊", "國", "顧", "é", "盤", "寳", "Λ", "龍", "鳳", "島", "誌", "緣", "結", "銭", "萬", "勝", "祎", "璟", "優", "歡", "臨", "時", "購", "＝", "★", "藍", "昇", "鐵", "觀", "勅", "農", "聲", "畫", "兿", "術", "發", "劉", "記", "專", "耑", "園", "書", "壴", "種", "Ο", "●", "褀", "號", "銀", "匯", "敟", "锘", "葉", "橪", "廣", "進", "蒄", "鑽", "阝", "祙", "貢", "鍋", "豊", "夬", "喆", "團", "閣", "開", "燁", "賓", "館", "酡", "沔", "順", "＋", "硚", "劵", "饸", "陽", "車", "湓", "復", "萊", "氣", "軒", "華", "堃", "迮", "纟", "戶", "馬", "學", "裡", "電", "嶽", "獨", "マ", "シ", "サ", "ジ", "燘", "袪", "環", "❤", "臺", "灣", "専", "賣", "孖", "聖", "攝", "線", "▪", "α", "傢", "俬", "夢", "達", "莊", "喬", "貝", "薩", "劍", "羅", "壓", "棛", "饦", "尃", "璈", "囍", "醫", "Ｇ", "Ｉ", "Ａ", "＃", "Ｎ", "鷄", "髙", "嬰", "啓", "約", "隹", "潔", "賴", "藝", "～", "寶", "籣", "麺", "　", "嶺", "√", "義", "網", "峩", "長", "∧", "魚", "機", "構", "②", "鳯", "偉", "Ｌ", "Ｂ", "㙟", "畵", "鴿", "＇", "詩", "溝", "嚞", "屌", "藔", "佧", "玥", "蘭", "織", "１", "３", "９", "０", "７", "點", "砭", "鴨", "鋪", "銘", "廳", "弍", "‧", "創", "湯", "坶", "℃", "卩", "骝", "＆", "烜", "荘", "當", "潤", "扞", "係", "懷", "碶", "钅", "蚨", "讠", "☆", "叢", "爲", "埗", "涫", "塗", "→", "楽", "現", "鯨", "愛", "瑪", "鈺", "忄", "悶", "藥", "飾", "樓", "視", "孬", "ㆍ", "燚", "苪", "師", "①", "丼", "锽", "│", "韓", "標", "è", "兒", "閏", "匋", "張", "漢", "Ü", "髪", "會", "閑", "檔", "習", "裝", "の", "峯", "菘", "輝", "И", "雞", "釣", "億", "浐", "Ｋ", "Ｏ", "Ｒ", "８", "Ｈ", "Ｅ", "Ｐ", "Ｔ", "Ｗ", "Ｄ", "Ｓ", "Ｃ", "Ｍ", "Ｆ", "姌", "饹", "»", "晞", "廰", "ä", "嵯", "鷹", "負", "飲", "絲", "冚", "楗", "澤", "綫", "區", "❋", "←", "質", "靑", "揚", "③", "滬", "統", "産", "協", "﹑", "乸", "畐", "經", "運", "際", "洺", "岽", "為", "粵", "諾", "崋", "豐", "碁", "ɔ", "Ｖ", "２", "６", "齋", "誠", "訂", "´", "勑", "雙", "陳", "無", "í", "泩", "媄", "夌", "刂", "ｉ", "ｃ", "ｔ", "ｏ", "ｒ", "ａ", "嘢", "耄", "燴", "暃", "壽", "媽", "靈", "抻", "體", "唻", "É", "冮", "甹", "鎮", "錦", "ʌ", "蜛", "蠄", "尓", "駕", "戀", "飬", "逹", "倫", "貴", "極", "Я", "Й", "寬", "磚", "嶪", "郎", "職", "｜", "間", "ｎ", "ｄ", "剎", "伈", "課", "飛", "橋", "瘊", "№", "譜", "骓", "圗", "滘", "縣", "粿", "咅", "養", "濤", "彳", "®", "％", "Ⅱ", "啰", "㴪", "見", "矞", "薬", "糁", "邨", "鲮", "顔", "罱", "З", "選", "話", "贏", "氪", "俵", "競", "瑩", "繡", "枱", "β", "綉", "á", "獅", "爾", "™", "麵", "戋", "淩", "徳", "個", "劇", "場", "務", "簡", "寵", "ｈ", "實", "膠", "轱", "圖", "築", "嘣", "樹", "㸃", "營", "耵", "孫", "饃", "鄺", "飯", "麯", "遠", "輸", "坫", "孃", "乚", "閃", "鏢", "㎡", "題", "廠", "關", "↑", "爺", "將", "軍", "連", "篦", "覌", "參", "箸", "－", "窠", "棽", "寕", "夀", "爰", "歐", "呙", "閥", "頡", "熱", "雎", "垟", "裟", "凬", "勁", "帑", "馕", "夆", "疌", "枼", "馮", "貨", "蒤", "樸", "彧", "旸", "靜", "龢", "暢", "㐱", "鳥", "珺", "鏡", "灡", "爭", "堷", "廚", "Ó", "騰", "診", "┅", "蘇", "褔", "凱", "頂", "豕", "亞", "帥", "嘬", "⊥", "仺", "桖", "複", "饣", "絡", "穂", "顏", "棟", "納", "▏", "濟", "親", "設", "計", "攵", "埌", "烺", "ò", "頤", "燦", "蓮", "撻", "節", "講", "濱", "濃", "娽", "洳", "朿", "燈", "鈴", "護", "膚", "铔", "過", "補", "Ｚ", "Ｕ", "５", "４", "坋", "闿", "䖝", "餘", "缐", "铞", "貿", "铪", "桼", "趙", "鍊", "［", "㐂", "垚", "菓", "揸", "捲", "鐘", "滏", "𣇉", "爍", "輪", "燜", "鴻", "鮮", "動", "鹞", "鷗", "丄", "慶", "鉌", "翥", "飮", "腸", "⇋", "漁", "覺", "來", "熘", "昴", "翏", "鲱", "圧", "鄉", "萭", "頔", "爐", "嫚", "г", "貭", "類", "聯", "幛", "輕", "訓", "鑒", "夋", "锨", "芃", "珣", "䝉", "扙", "嵐", "銷", "處", "ㄱ", "語", "誘", "苝", "歸", "儀", "燒", "楿", "內", "粢", "葒", "奧", "麥", "礻", "滿", "蠔", "穵", "瞭", "態", "鱬", "榞", "硂", "鄭", "黃", "煙", "祐", "奓", "逺", "＊", "瑄", "獲", "聞", "薦", "讀", "這", "樣", "決", "問", "啟", "們", "執", "説", "轉", "單", "隨", "唘", "帶", "倉", "庫", "還", "贈", "尙", "皺", "■", "餅", "產", "○", "∈", "報", "狀", "楓", "賠", "琯", "嗮", "禮", "｀", "傳", "＞", "≤", "嗞", "Φ", "≥", "換", "咭", "∣", "↓", "曬", "ε", "応", "寫", "″", "終", "様", "純", "費", "療", "聨", "凍", "壐", "郵", "ü", "黒", "∫", "製", "塊", "調", "軽", "確", "撃", "級", "馴", "Ⅲ", "涇", "繹", "數", "碼", "證", "狒", "処", "劑", "＜", "晧", "賀", "衆", "］", "櫥", "兩", "陰", "絶", "對", "鯉", "憶", "◎", "ｐ", "ｅ", "Ｙ", "蕒", "煖", "頓", "測", "試", "鼽", "僑", "碩", "妝", "帯", "≈", "鐡", "舖", "權", "喫", "倆", "ˋ", "該", "悅", "ā", "俫", "．", "ｆ", "ｓ", "ｂ", "ｍ", "ｋ", "ｇ", "ｕ", "ｊ", "貼", "淨", "濕", "針", "適", "備", "ｌ", "／", "給", "謢", "強", "觸", "衛", "與", "⊙", "＄", "緯", "變", "⑴", "⑵", "⑶", "㎏", "殺", "∩", "幚", "─", "價", "▲", "離", "ú", "ó", "飄", "烏", "関", "閟", "﹝", "﹞", "邏", "輯", "鍵", "驗", "訣", "導", "歷", "屆", "層", "▼", "儱", "錄", "熳", "ē", "艦", "吋", "錶", "辧", "飼", "顯", "④", "禦", "販", "気", "対", "枰", "閩", "紀", "幹", "瞓", "貊", "淚", "△", "眞", "墊", "Ω", "獻", "褲", "縫", "緑", "亜", "鉅", "餠", "｛", "｝", "◆", "蘆", "薈", "█", "◇", "溫", "彈", "晳", "粧", "犸", "穩", "訊", "崬", "凖", "熥", "П", "舊", "條", "紋", "圍", "Ⅳ", "筆", "尷", "難", "雜", "錯", "綁", "識", "頰", "鎖", "艶", "□", "殁", "殼", "⑧", "├", "▕", "鵬", "ǐ", "ō", "ǒ", "糝", "綱", "▎", "μ", "盜", "饅", "醬", "籤", "蓋", "釀", "鹽", "據", "à", "ɡ", "辦", "◥", "彐", "┌", "婦", "獸", "鲩", "伱", "ī", "蒟", "蒻", "齊", "袆", "腦", "寧", "凈", "妳", "煥", "詢", "偽", "謹", "啫", "鯽", "騷", "鱸", "損", "傷", "鎻", "髮", "買", "冏", "儥", "両", "﹢", "∞", "載", "喰", "ｚ", "羙", "悵", "燙", "曉", "員", "組", "徹", "艷", "痠", "鋼", "鼙", "縮", "細", "嚒", "爯", "≠", "維", "＂", "鱻", "壇", "厍", "帰", "浥", "犇", "薡", "軎", "²", "應", "醜", "刪", "緻", "鶴", "賜", "噁", "軌", "尨", "镔", "鷺", "槗", "彌", "葚", "濛", "請", "溇", "緹", "賢", "訪", "獴", "瑅", "資", "縤", "陣", "蕟", "栢", "韻", "祼", "恁", "伢", "謝", "劃", "涑", "總", "衖", "踺", "砋", "凉", "籃", "駿", "苼", "瘋", "昽", "紡", "驊", "腎", "﹗", "響", "杋", "剛", "嚴", "禪", "歓", "槍", "傘", "檸", "檫", "炣", "勢", "鏜", "鎢", "銑", "尐", "減", "奪", "惡", "θ", "僮", "婭", "臘", "ū", "ì", "殻", "鉄", "∑", "蛲", "焼", "緖", "續", "紹", "懮"};*/
    return CHINESE;
}

std::vector<std::string> get_english()
{
    std::vector<std::string> ENGLISH = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"};
    return ENGLISH;
}

