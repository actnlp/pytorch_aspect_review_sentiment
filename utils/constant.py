PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

ASP_TO_ID = {'航班准点': 0, '航班体验': 1, '空乘服务': 2, '机上餐食口味': 3, '飞行平稳': 4, '无关评论': 5, '机上餐食种类': 6, '机上娱乐设备': 7, '机上餐食分量': 8, '机龄': 9, '机上空间': 10, '机上座椅': 11, '机上广播': 12, '空乘颜值': 13, '机型': 14, '机上设施': 15, '到达机位': 16, '延误处理': 17, '机票价格': 18, '机场体验': 19, '机上卫生': 20, '行李服务': 21, '登机机位': 22, '机上噪音': 23, '行李提取时间': 24, '机上饮品': 25, '值机服务': 26, '机场地勤服务': 27, '机上温度': 28, '行李托运': 29, '登机口管理': 30, '会员权益': 31, '脏话': 32, '机场餐饮质量': 33, '下客服务': 34, 'APP值机': 35, '滑行时长': 36, '安检体验': 37, '火星语': 38, '旅客管理': 39, '中转时间': 40, '安全管理': 41, '转机服务': 42, '官网APP': 43, '广告': 44, '机场餐饮种类': 45, '机上杂志': 46, '机场地面设施': 47, '机上公共节目': 48, '贵宾厅服务': 49, '涉政': 50, '机场大巴等待时间': 51, '退改签服务': 52, '安检质量': 53, '贵宾厅餐食': 54, '贵宾厅设施': 55, '候机厅空间': 56, '接送机': 57, '保险': 58, '候机厅商业': 59, 'APP选餐': 60, '机场大巴速度': 61, '机场餐饮价格': 62}


ASP_TOKEN = ['颜值', '机票价格', '安检', '飞行', '节目', '设施', '长', '空间', '服务', '选餐', '准点', '广播', '中转', '会员', '质量', '平稳', '机龄', '权益', '娱乐', '机上', '空乘', '地面', '贵宾厅', '餐食', '涉政', '卫生', '机场', '价格', '口味', '分量', '登机', '延误', '候机厅', '托运', 'APP', '航班', '餐饮', '体验', '无关', '机型', '广告', '官网', '滑行', '转机', '地勤', '机位', '登机口', '提取', '到达', '种类', '时间', '管理', '公共', '饮品', '时', '值机', '座椅', '处理', '火星', '杂志', '退', '语', '等待时间', '设备', '温度', '噪音', '行李', '商业', '改签', '下客', '脏话', '评论', '大巴']

LABEL_TO_ID = {'-1': 0, '0': 1, '1': 2, '-2': 3}

