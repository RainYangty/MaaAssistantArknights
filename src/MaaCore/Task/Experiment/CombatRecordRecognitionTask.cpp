#include "CombatRecordRecognitionTask.h"

#include <ranges>
#include <unordered_map>
#include <unordered_set>

#include "Config/Miscellaneous/BattleDataConfig.h"
#include "Config/Miscellaneous/TilePack.h"
#include "Config/TaskData.h"
#include "MaaUtils/ImageIo.h"
#include "MaaUtils/NoWarningCV.hpp"
#include "Utils/Logger.hpp"
#include "Vision/Battle/BattleFormationAnalyzer.h"
#include "Vision/Battle/BattlefieldClassifier.h"
#include "Vision/Battle/BattlefieldDetector.h"
#include "Vision/Battle/BattlefieldMatcher.h"
#include "Vision/BestMatcher.h"
#include "Vision/RegionOCRer.h"
#include <opencv2/core/ocl.hpp>

// #define Mat UMat

bool asst::CombatRecordRecognitionTask::set_video_path(const std::filesystem::path& path)
{
    if (!std::filesystem::exists(path)) {
        Log.error(__FUNCTION__, "filename not exists", path);
        return false;
    }
    m_video_path = path;
    return true;
}

bool asst::CombatRecordRecognitionTask::_run()
{
    LogTraceFunction;

    cv::ocl::setUseOpenCL(true);

    auto release_video = [](cv::VideoCapture* video) {
        if (video && video->isOpened()) {
            video->release();
        }
    };
    auto crt_path = utils::path_to_crt_string(m_video_path);
    m_video_ptr = std::shared_ptr<cv::VideoCapture>(new cv::VideoCapture(crt_path), release_video);

    if (!m_video_ptr->isOpened()) {
        Log.error(__FUNCTION__, "video_io open failed", m_video_path);
        return false;
    }
    m_video_fps = m_video_ptr->get(cv::CAP_PROP_FPS);
    m_video_frame_count = static_cast<size_t>(m_video_ptr->get(cv::CAP_PROP_FRAME_COUNT));
    m_battle_start_frame = 0;
    // m_scale = WindowHeightDefault / m_video_ptr->get(cv::CAP_PROP_FRAME_HEIGHT);
    double raw_w = m_video_ptr->get(cv::CAP_PROP_FRAME_WIDTH);
    double raw_h = m_video_ptr->get(cv::CAP_PROP_FRAME_HEIGHT);
    const double target_ratio = 1280.0 / 720.0;
    current_ratio = raw_w / raw_h;
    m_scale = WindowWidthDefault / raw_w;
    /*
     用于地图定位，方舟的地图具有如下规律
     1. 不论长宽，哪边过大就对齐另外一边，比如21:9就将高度锁定在720，9:21，就把宽度锁定在1280
     2.锁定后，锁定的边ROI坐标不用变，另一边用中心减去对应坐标
     如2746*1908的就锁定宽（因为高过高），成为1280*889.38，那么获取的坐标比如是(a,b)a是水平，b是竖直，那么映射之后应该是(a,
     889.38/2+b-720/2)
    */
    if (current_ratio > target_ratio) {
        // 若锁定高度为 720
        is_height_locked = true;
        dectect_scale = 720.0 / raw_h;
        m_offset_x = (raw_w * m_scale - 1280.0) / 2.0;
        m_offset_y = 0;
    }
    else {
        // 若锁定宽度为 1280
        dectect_scale = m_scale;
        m_offset_x = 0;
        m_offset_y = (raw_h * m_scale - 720.0) / 2.0;
    }

    Log.info("Adaptive Scaling | Scale:", m_scale, "Offset_X:", m_offset_x, "Offset_Y:", m_offset_y);

    if (!analyze_formation()) {
        Log.error(__FUNCTION__, "failed to analyze formation");
        return false;
    }

    if (!analyze_stage()) {
        Log.error(__FUNCTION__, "unknown stage");
        return false;
    }

    if (!analyze_deployment()) {
        Log.error(__FUNCTION__, "failed to match deployment");
        return false;
    }

    if (!slice_video()) {
        Log.error(__FUNCTION__, "failed to slice");
        return false;
    }

    ClipInfo* pre_valid = nullptr;
    for (auto iter = m_clips.begin(); iter != m_clips.end(); ++iter) {
        auto& clip = *iter;

        if (!clip.deployment_changed && iter != m_clips.begin()) {
            compare_skill(clip, *(iter - 1));
            continue;
        }

        if (!analyze_clip(clip, pre_valid)) {
            Log.error(__FUNCTION__, "failed to analyze clip");
            return false;
        }
        pre_valid = &clip;
    }

    Log.info("full copilot json", m_copilot_json.to_string());

    std::string filename = std::format(
        "MaaAI_{}_{}_{}.json",
        m_stage_name,
        utils::path_to_utf8_string(m_video_path.stem()),
        MAA_NS::format_now_for_filename());
    auto filepath = UserDir.get() / "cache" / "CombatRecord" / utils::path(filename);
    std::filesystem::create_directories(filepath.parent_path());
    std::ofstream osf(filepath);
    osf << m_copilot_json.format();
    osf.close();

    auto cb_json = basic_info_with_what("Finished");
    cb_json["details"]["filename"] = utils::path_to_utf8_string(filepath);
    callback(AsstMsg::SubTaskExtraInfo, cb_json);

    return true;
}

bool asst::CombatRecordRecognitionTask::analyze_formation()
{
    LogTraceFunction;
    callback(AsstMsg::SubTaskStart, basic_info_with_what("OcrFormation"));

    const int skip_count = m_video_fps > m_formation_fps ? static_cast<int>(m_video_fps / m_formation_fps) - 1 : 0;

    BattleFormationAnalyzer formation_ananlyzer;
    int no_changes_count = 0;
    for (size_t i = 0; i < m_video_frame_count; i += skip_frames(skip_count) + 1) {
        cv::Mat frame;
        *m_video_ptr >> frame;
        if (frame.empty()) {
            Log.error(i, "frame is empty");
            callback(AsstMsg::SubTaskError, basic_info_with_what("OcrFormation"));
            return false;
        }

        // cv::resize(frame, frame, cv::Size(), m_scale, m_scale, cv::INTER_AREA);
        int raw_w = frame.cols;
        int raw_h = frame.rows;

        double target_ratio = 1280.0 / 720.0;
        int crop_x, crop_y, crop_w, crop_h;

        if (current_ratio > target_ratio) {
            // 宽屏的宽度方向需要居中裁掉两边
            crop_h = raw_h;
            crop_w = static_cast<int>(std::round(raw_h * target_ratio));
            crop_x = (raw_w - crop_w) / 2;
            crop_y = 0;
        }
        else {
            // 窄屏的高度方向需要居中裁掉上下
            crop_w = raw_w;
            crop_h = static_cast<int>(std::round(720.0 / m_scale));
            crop_x = 0;
            crop_y = static_cast<int>(std::round(m_offset_y / m_scale));
        }

        // 防止浮点数舍入导致越界 1 像素
        crop_x = std::clamp(crop_x, 0, raw_w - 1);
        crop_y = std::clamp(crop_y, 0, raw_h - 1);
        crop_w = std::clamp(crop_w, 1, raw_w - crop_x);
        crop_h = std::clamp(crop_h, 1, raw_h - crop_y);

        cv::Rect crop_roi(crop_x, crop_y, crop_w, crop_h);

        frame = frame(crop_roi);
        cv::resize(frame, frame, cv::Size(1280, 720), 0, 0, cv::INTER_AREA);

        formation_ananlyzer.set_image(frame);
        auto formation_opt = formation_ananlyzer.analyze();
        // 有些视频会有个过渡或者动画啥的，只取一帧识别的可能不全。多识别几帧
        if (formation_opt) {
            if (formation_opt->size() > m_formation.size()) {
                for (const auto& [name, avatar] : *formation_opt) {
                    m_formation.insert_or_assign(name, avatar);
                }
            }
            else if (++no_changes_count > 5) {
                m_formation_end_frame = i;
                break;
            }
        }
        else if (!m_formation.empty()) {
            m_formation_end_frame = i;
            break;
        }
    }

    Log.info("Formation:", m_formation | std::views::keys);
    auto cb_info = basic_info_with_what("OcrFormation");
    auto& cb_formation = cb_info["details"]["formation"];
    for (const auto& [name, avatar] : m_formation) {
        std::vector<battle::OperUsage> opers;
        opers.emplace_back(battle::OperUsage { name, 0, battle::SkillUsage::NotUse });
        json::object oper_json { { "name", name }, { "skill", 1 }, { "skill_usage", 0 } };
        m_copilot_json["opers"].emplace(std::move(oper_json));

        cb_formation.emplace(name);
        MAA_NS::imwrite(utils::path("debug/video_export/formation/") / utils::path(name + ".png"), avatar);
    }
    callback(AsstMsg::SubTaskCompleted, cb_info);

    return true;
}

bool asst::CombatRecordRecognitionTask::analyze_stage()
{
    LogTraceFunction;

    callback(AsstMsg::SubTaskStart, basic_info_with_what("OcrStage"));

    const auto stage_name_task_ptr = Task.get("BattleStageName");
    const int skip_count = m_video_fps > m_stage_ocr_fps ? static_cast<int>(m_video_fps / m_stage_ocr_fps) - 1 : 0;

    for (size_t i = m_formation_end_frame; i < m_video_frame_count; i += skip_frames(skip_count) + 1) {
        cv::Mat frame;
        *m_video_ptr >> frame;
        if (frame.empty()) {
            Log.error(i, "frame is empty");
            callback(AsstMsg::SubTaskError, basic_info_with_what("OcrStage"));
            return false;
        }

        // cv::Rect ui_roi(static_cast<int>(m_offset_x), static_cast<int>(m_offset_y), 1280, 720);

        cv::resize(frame, frame, cv::Size(), m_scale, m_scale, cv::INTER_AREA);

        cv::Mat standard_frame = cv::Mat::zeros(720, 1280, frame.type());

        int src_x = static_cast<int>(std::round(m_offset_x));
        int src_y = static_cast<int>(std::round(m_offset_y));

        int src_w = std::min(1280, frame.cols - src_x);
        int src_h = std::min(720, frame.rows - src_y);

        cv::Rect src_roi(src_x, src_y, src_w, src_h);

        int dst_x = (1280 - src_w) / 2;
        int dst_y = (720 - src_h) / 2;

        cv::Rect dst_roi(dst_x, dst_y, src_w, src_h);

        frame(src_roi).copyTo(standard_frame(dst_roi));

        frame = standard_frame;

        RegionOCRer stage_analyzer(frame);
        stage_analyzer.set_task_info(stage_name_task_ptr);
        bool analyzed = stage_analyzer.analyze().has_value();

        if (!analyzed) {
            // BattlefieldMatcher battle_analyzer(frame);
            // if (battle_analyzer.analyze()) {
            //     Log.error(i, "already start button, but still failed to analyze stage name");
            //     m_stage_ocr_end_frame = i;
            //     callback(AsstMsg::SubTaskError, basic_info_with_what("OcrStage"));
            //     return false;
            // }
            continue;
        }
        const std::string& text = stage_analyzer.get_result().text;

        if (text.empty() || !Tile.find(text)) {
            continue;
        }

        m_stage_name = text;
        m_stage_ocr_end_frame = i;
        break;
    }

    Log.info("Stage", m_stage_name);
    if (m_stage_name.empty() || !Tile.find(m_stage_name)) {
        callback(AsstMsg::SubTaskError, basic_info_with_what("OcrStage"));
        return false;
    }
    auto calc_result = Tile.calc(m_stage_name);
    m_normal_tile_info = std::move(calc_result.normal_tile_info);

    m_copilot_json["stage_name"] = m_stage_name;
    m_copilot_json["minimum_required"] = "v4.0.0";
    m_copilot_json["doc"]["title"] = "MAA AI - " + m_stage_name;
    m_copilot_json["doc"]["details"] =
        "Built at: " + MAA_NS::format_now() + "\n" + utils::path_to_utf8_string(m_video_path);

    callback(AsstMsg::SubTaskCompleted, basic_info_with_what("OcrStage"));
    return true;
}

bool asst::CombatRecordRecognitionTask::analyze_deployment()
{
    LogTraceFunction;
    callback(AsstMsg::SubTaskStart, basic_info_with_what("MatchDeployment"));

    const int skip_count = m_video_fps > m_deployment_fps ? static_cast<int>(m_video_fps / m_deployment_fps) - 1 : 0;

    BattlefieldMatcher oper_analyzer;
    oper_analyzer.set_object_of_interest({ .deployment = true });

    std::vector<battle::DeploymentOper> deployment;

    // double x_compress_factor = 1.0;
    for (size_t i = m_stage_ocr_end_frame; i < m_video_frame_count; i += skip_frames(skip_count) + 1) {
        cv::Mat frame;
        *m_video_ptr >> frame;
        if (frame.empty()) {
            Log.error(i, "frame is empty");
            callback(AsstMsg::SubTaskError, basic_info_with_what("MatchDeployment"));
            return false;
        }

        cv::Mat frame_pause = frame.clone();

        cv::resize(frame_pause, frame_pause, cv::Size(), m_scale, m_scale, cv::INTER_AREA);

        cv::Rect ui_roi(0, 0, 1280, std::min(720, frame_pause.rows));
        frame_pause = frame_pause(ui_roi);

        oper_analyzer.set_image(frame_pause);
        auto oper_result_opt = oper_analyzer.analyze();
        bool analyzed = oper_result_opt && oper_result_opt->pause_button;
        if (analyzed) {
            m_battle_start_frame = i;

            // 自适应比例的干员识别
            // double current_ratio = static_cast<double>(frame.cols) / frame.rows;
            double target_ratio = 1280.0 / 720.0;

            if (current_ratio <= target_ratio) {
                // 窄屏 按宽度 1280 等比缩放，然后从下往上截取 720 高度
                double scale_1280 = 1280.0 / frame.cols;
                cv::Mat narrow_standard_frame;
                cv::resize(frame, narrow_standard_frame, cv::Size(), scale_1280, scale_1280, cv::INTER_AREA);

                int crop_y = std::max(0, narrow_standard_frame.rows - 720);
                cv::Rect bottom_roi(0, crop_y, 1280, std::min(720, narrow_standard_frame.rows));
                cv::Mat final_frame = narrow_standard_frame(bottom_roi);

                oper_analyzer.set_image(final_frame);
                auto final_oper_opt = oper_analyzer.analyze();
                if (final_oper_opt) {
                    deployment = std::move(final_oper_opt->deployment);
                }
            }
            else {
                // 宽屏 按高度 720 等比缩放
                double scale_720 = 720.0 / frame.rows;
                cv::Mat wide_standard_frame;
                cv::resize(frame, wide_standard_frame, cv::Size(), scale_720, scale_720, cv::INTER_AREA);

                int width = wide_standard_frame.cols;

                // 宽屏：切割为左右两段 1280x720 进行检测
                cv::Rect left_roi(0, 0, 1280, 720);
                int right_offset_x = width - 1280;
                cv::Rect right_roi(right_offset_x, 0, 1280, 720);

                // 识别左半边
                oper_analyzer.set_image(wide_standard_frame(left_roi));
                std::vector<battle::DeploymentOper> left_opers;
                if (auto left_opt = oper_analyzer.analyze()) {
                    left_opers = std::move(left_opt->deployment);
                }

                // 识别右半边
                oper_analyzer.set_image(wide_standard_frame(right_roi));
                std::vector<battle::DeploymentOper> right_opers;
                if (auto right_opt = oper_analyzer.analyze()) {
                    right_opers = std::move(right_opt->deployment);
                }

                // 将右半边的干员坐标还原到宽图的绝对坐标系
                for (auto& oper : right_opers) {
                    oper.rect.x += right_offset_x;
                }

                // 去重并合并
                deployment = std::move(left_opers);
                const int duplicate_threshold_x = 50; // 同一干员X坐标偏差容忍度

                for (auto& r_oper : right_opers) {
                    bool is_duplicate = false;
                    for (const auto& l_oper : deployment) {
                        if (std::abs(r_oper.rect.x - l_oper.rect.x) < duplicate_threshold_x) {
                            is_duplicate = true;
                            break;
                        }
                    }
                    if (!is_duplicate) {
                        deployment.push_back(std::move(r_oper));
                    }
                }

                // 重新从左到右排序
                std::sort(
                    deployment.begin(),
                    deployment.end(),
                    [](const battle::DeploymentOper& a, const battle::DeploymentOper& b) {
                        return a.rect.x < b.rect.x;
                    });

                // 重建正确的索引
                size_t new_index = 0;
                for (auto& oper : deployment) {
                    oper.index = new_index++;
                }
            }

            break; // 成功解析出开始帧和 deployment，跳出循环
        }
    }

    auto avatar_task_ptr = Task.get("BattleAvatarDataForFormation");
    for (const auto& [name, formation_avatar] : m_formation) {
        cv::Mat target_formation_avatar = formation_avatar.clone();

        BestMatcher best_match_analyzer(target_formation_avatar);
        best_match_analyzer.set_task_info(avatar_task_ptr);

        std::unordered_set<battle::Role> roles = { BattleData.get_role(name) };
        if (name == "阿米娅") {
            roles.emplace(battle::Role::Warrior);
        }

        // 编队界面，有些视频会有些花里胡哨的特效遮挡啥的，所以尽量减小点模板尺寸
        auto crop_roi = make_rect<cv::Rect>(avatar_task_ptr->rect_move);
        // 小车的缩放太离谱了
        const size_t scale_ends = BattleData.get_rarity(name) == 1 ? 200 : 125;
        std::unordered_map<std::string, cv::Mat> candidate;
        for (const auto& oper : deployment) {
            if (!roles.contains(oper.role)) {
                continue;
            }
            cv::Mat crop_avatar = oper.avatar(crop_roi);
            // 从编队到待部署区，每个干员的缩放大小都不一样，暴力跑一遍
            // TODO: 不知道gamedata里有没有这个缩放数据，直接去拿
            // 这里的 resized_avatar 依然是被压扁的状态，刚好拿去和压扁的 target_formation_avatar 匹配
            for (size_t i = 100; i < scale_ends; ++i) {
                double avatar_scale = i / 100.0;
                const auto resize_method = avatar_scale < 1.0 ? cv::INTER_AREA : cv::INTER_LINEAR;
                cv::Mat resized_avatar;
                cv::resize(crop_avatar, resized_avatar, cv::Size(), avatar_scale, avatar_scale, resize_method);
                std::string flag = name + "|" + std::to_string(oper.index) + "|" + std::to_string(i);
                best_match_analyzer.append_templ(flag, resized_avatar);
                candidate.emplace(flag, oper.avatar);
            }
        }
        bool analyzed = best_match_analyzer.analyze().has_value();
        if (!analyzed) {
            Log.warn(m_battle_start_frame, "failed to match", name);
            continue;
        }
        m_all_avatars.emplace(name, candidate.at(best_match_analyzer.get_result().templ_info.name));
    }
    callback(AsstMsg::SubTaskCompleted, basic_info_with_what("MatchDeployment"));

    return !m_all_avatars.empty();
}

bool asst::CombatRecordRecognitionTask::slice_video()
{
    LogTraceFunction;
    callback(AsstMsg::SubTaskStart, basic_info_with_what("Slice"));

    const int skip_count = m_video_fps > m_deployment_fps ? static_cast<int>(m_video_fps / m_deployment_fps) - 1 : 0;

    int not_in_battle_count = 0;
    bool in_segment = false;

    cv::Mat frame;
    cv::Mat pre_frame;

    size_t i = m_battle_start_frame;
    size_t ends = m_video_frame_count - skip_count - 10;

    int latest_kills = -1;
    int total_kills = -1;

    auto battle_over = [&]() {
        if (m_clips.empty()) {
            return;
        }
        if (m_battle_end_frame == 0) {
            m_battle_end_frame = i;
        }
        if (!in_segment) {
            return;
        }
        auto& pre_clip = m_clips.back();
        pre_clip.end_frame_index = i - skip_count;
        pre_clip.end_frame = pre_frame;
        in_segment = false;
    };
    for (; i < ends; i += skip_frames(skip_count) + 1, pre_frame = frame) {
        cv::Mat temp;
        *m_video_ptr >> temp;
        frame = temp;
        if (frame.empty()) {
            Log.warn(i, "frame is empty");
            battle_over();
            break;
        }

        // 使用拼图识别击杀数、按钮
        cv::Mat standard_frame = get_stitched_720p(frame);
        BattlefieldMatcher ui_analyzer(standard_frame);
        ui_analyzer.set_object_of_interest({ .kills = true, .speed_button = true });
        ui_analyzer.set_total_kills_prompt(total_kills);
        auto ui_result_opt = ui_analyzer.analyze();

        if (!ui_result_opt) {
            battle_over();
            if (++not_in_battle_count > 10) {
                break;
            }
            continue;
        }
        m_battle_end_frame = 0;
        not_in_battle_count = 0;

        // --- 2. 干员轨道：使用自适应逻辑识别底部栏 ---
        std::vector<battle::DeploymentOper> cur_opers;
        BattlefieldMatcher oper_analyzer;
        oper_analyzer.set_object_of_interest({ .deployment = true });

        double target_ratio = 1280.0 / 720.0;
        if (current_ratio <= target_ratio)
        {
            // 【窄屏】从下往上截取识别
            double scale_1280 = 1280.0 / frame.cols;
            cv::Mat narrow_standard_frame;
            cv::resize(frame, narrow_standard_frame, cv::Size(), scale_1280, scale_1280, cv::INTER_AREA);

            int crop_y = std::max(0, narrow_standard_frame.rows - 720);
            cv::Rect bottom_roi(0, crop_y, 1280, std::min(720, narrow_standard_frame.rows));

            oper_analyzer.set_image(narrow_standard_frame(bottom_roi));
            if (auto opt = oper_analyzer.analyze()) {
                cur_opers = std::move(opt->deployment);
            }
        }
        else
        {
            // 【宽屏】分段滑动窗口识别
            double scale_720 = 720.0 / frame.rows;
            cv::Mat wide_standard_frame;
            cv::resize(frame, wide_standard_frame, cv::Size(), scale_720, scale_720, cv::INTER_AREA);

            int width = wide_standard_frame.cols;
            cv::Rect left_roi(0, 0, 1280, 720);
            int right_offset_x = width - 1280;
            cv::Rect right_roi(right_offset_x, 0, 1280, 720);

            // 左半边
            oper_analyzer.set_image(wide_standard_frame(left_roi));
            if (auto left_opt = oper_analyzer.analyze()) {
                cur_opers = std::move(left_opt->deployment);
            }
            // 右半边
            oper_analyzer.set_image(wide_standard_frame(right_roi));
            if (auto right_opt = oper_analyzer.analyze()) {
                auto right_opers = std::move(right_opt->deployment);
                // 坐标还原与合并去重
                for (auto& oper : right_opers) {
                    oper.rect.x += right_offset_x;
                    bool is_duplicate = false;
                    for (const auto& l_oper : cur_opers) {
                        if (std::abs(oper.rect.x - l_oper.rect.x) < 50) {
                            is_duplicate = true;
                            break;
                        }
                    }
                    if (!is_duplicate) {
                        cur_opers.push_back(std::move(oper));
                    }
                }
            }
            // 排序与索引重构
            std::sort(cur_opers.begin(), cur_opers.end(), [](const auto& a, const auto& b) {
                return a.rect.x < b.rect.x;
            });
            for (size_t idx = 0; idx < cur_opers.size(); ++idx) {
                cur_opers[idx].index = idx;
            }
        }

        // 结合两路识别结果

        // 更新击杀数
        if (ui_result_opt->kills.status == BattlefieldMatcher::MatchStatus::Success) {
            auto& [cur_kills, cur_total_kills] = ui_result_opt->kills.value;
            if (cur_kills != latest_kills) {
                m_frame_kills.emplace_back(std::make_pair(i, cur_kills));
            }
            latest_kills = cur_kills;
            total_kills = cur_total_kills;
        }

        // 连贯性校验
        bool continuity = true;
        int pre_distance = 0;
        for (size_t idx = 1; idx < cur_opers.size(); ++idx) {
            int distance = std::abs(cur_opers[idx].rect.x - cur_opers[idx - 1].rect.x);
            if (pre_distance && std::abs(distance - pre_distance) > 5) {
                continuity = false;
                break;
            }
            pre_distance = distance;
        }

        // 状态判定
        bool oper_is_clicked = !ui_result_opt->speed_button || !ui_result_opt->pause_button;
        bool oper_auto_retreat =
            in_segment && continuity && !m_clips.empty() && cur_opers.size() != m_clips.back().deployment.size();

        if (oper_is_clicked || oper_auto_retreat) {
            if (m_clips.empty()) {
                continue;
            }
            auto& pre_clip = m_clips.back();
            if (pre_clip.ends_oper_name.empty()) {
                pre_clip.ends_oper_name = analyze_detail_page_oper_name(frame);
            }
            if (!in_segment) {
                continue;
            }
            pre_clip.end_frame_index = i - skip_count;
            pre_clip.end_frame = pre_frame;
            in_segment = false;
            continue;
        }
        else if (!continuity) {
            Log.warn(i, "opers is not continuity");
            continue;
        }
        else if (!in_segment) {
            ClipInfo info;
            info.start_frame_index = i;
            info.end_frame_index = i;
            info.deployment = cur_opers; // 存入精准的干员列表
            info.start_frame = frame;
            m_clips.emplace_back(std::move(info));
            in_segment = true;
        }
    }
    battle_over();

    if (m_clips.empty()) {
        callback(AsstMsg::SubTaskError, basic_info_with_what("Slice"));
        return false;
    }

    // post process clips
    constexpr int offset_ms = 300;
    const int offset_frame = static_cast<int>(offset_ms * m_video_fps / 1000.0);

    for (auto iter = m_clips.begin(); iter != m_clips.end();) {
        ClipInfo& clip = *iter;
        if (clip.end_frame_index <= clip.start_frame_index) {
            Log.warn(
                __FUNCTION__,
                "deployment has no changes or frame error",
                clip.start_frame_index,
                clip.end_frame_index);
            iter = m_clips.erase(iter);
            continue;
        }

        size_t new_start = clip.start_frame_index + offset_frame;
        if (new_start < clip.end_frame_index) {
            clip.start_frame_index = new_start;
        }

        if (iter == m_clips.begin()) {
            ++iter;
            continue;
        }

        ClipInfo& pre_clip = *(iter - 1);
        bool deployment_changed = false;
        if (iter != m_clips.begin() && clip.deployment.size() == pre_clip.deployment.size()) {
            for (size_t j = 0; j < clip.deployment.size(); ++j) {
                deployment_changed |= clip.deployment[j].role != pre_clip.deployment[j].role;
            }
        }
        else {
            deployment_changed = true;
        }
        iter->deployment_changed = deployment_changed;
        ++iter;
    }
    m_video_ptr->set(cv::CAP_PROP_POS_FRAMES, static_cast<double>(m_battle_start_frame));

    callback(AsstMsg::SubTaskCompleted, basic_info_with_what("Slice"));
    return true;
}

bool asst::CombatRecordRecognitionTask::analyze_clip(ClipInfo& clip, ClipInfo* pre_clip_ptr)
{
    LogTraceFunction;

    if (!detect_operators(clip, pre_clip_ptr)) {
        return false;
    }
    if (!classify_direction(clip, pre_clip_ptr)) {
        return false;
    }
    if (!process_changes(clip, pre_clip_ptr)) {
        return false;
    }

    return true;
}

bool asst::CombatRecordRecognitionTask::compare_skill(ClipInfo& clip, ClipInfo& pre_clip)
{
    LogTraceFunction;

    callback(AsstMsg::SubTaskStart, basic_info_with_what("CompSkill"));

    const std::string oper_name = pre_clip.ends_oper_name;
    const Point target_location = m_operator_locations[oper_name];
    const Point target_position = m_normal_tile_info[target_location].pos;
    BattlefieldClassifier analyzer(pre_clip.end_frame);
    analyzer.set_object_of_interest({ .skill_ready = true });
    analyzer.set_base_point(target_position);
    bool pre_ready = analyzer.analyze()->skill_ready.ready;

    if (!pre_ready) {
        // TODO: 有可能是点开之后等着技能转好，这种情况比较难处理
        return true;
    }

    // 有些干员明明点完技能了，但图标还会持续几百毫秒才消失
    // 所以这里要跳过一段时间
    constexpr int skip_ms = 500;
    size_t cls_begin = clip.start_frame_index + static_cast<size_t>(skip_ms * m_video_fps / 1000.0);
    if (cls_begin > clip.end_frame_index) {
        Log.warn("skip too much");
        cls_begin = clip.start_frame_index + (clip.end_frame_index - clip.start_frame_index) / 2;
    }
    const size_t begin_skip = cls_begin - static_cast<size_t>(m_video_ptr->get(cv::CAP_PROP_POS_FRAMES));
    skip_frames(begin_skip);

    cv::Mat frame;
    *m_video_ptr >> frame;
    if (frame.empty()) {
        Log.error("frame is empty");
        callback(AsstMsg::SubTaskError, basic_info_with_what("CompSkill"));
        return false;
    }
    cv::resize(frame, frame, cv::Size(), m_scale, m_scale, cv::INTER_AREA);

    double h_current = frame.rows;
    int mapped_y = static_cast<int>(std::round(h_current / 2.0 + target_position.y - 360.0));

    Point actual_target_point { target_position.x, mapped_y };

    analyzer.set_image(frame);
    analyzer.set_base_point(actual_target_point);
    bool cur_ready = analyzer.analyze()->skill_ready.ready;

    if (pre_ready && !cur_ready) {
        json::object condition = analyze_action_condition(clip, &pre_clip);
        json::object skill_json = condition | json::object {
            { "type", "Skill" },
            { "location", json::array { target_location.x, target_location.y } },
            { "name", oper_name },
        };
        Log.info("skill json", skill_json.to_string());
        auto& actions_json = m_copilot_json["actions"].as_array();
        actions_json.emplace_back(std::move(skill_json));
    }

    callback(AsstMsg::SubTaskCompleted, basic_info_with_what("CompSkill"));
    return true;
}

bool asst::CombatRecordRecognitionTask::detect_operators(ClipInfo& clip, [[maybe_unused]] ClipInfo* pre_clip_ptr)
{
    LogTraceFunction;

    callback(AsstMsg::SubTaskStart, basic_info_with_what("DetectOperators"));

    const size_t frame_count = clip.end_frame_index - clip.start_frame_index;

    /* detect operators on the battefield */
    using DetectionResult = std::unordered_set<Point>;
    std::unordered_map<DetectionResult, size_t, ContainerHasher<DetectionResult>> oper_det_samping;
    const Rect det_box_move = Task.get("BattleOperBoxRectMove")->rect_move;

    constexpr size_t OperDetSamplingCount = 20;
    const size_t skip_count =
        frame_count > (OperDetSamplingCount + 1) ? frame_count / (OperDetSamplingCount + 1) - 1 : 0;

    const size_t det_begin = clip.start_frame_index + skip_count;
    const size_t det_end = clip.end_frame_index - skip_count;

    const size_t begin_skip = det_begin - static_cast<size_t>(m_video_ptr->get(cv::CAP_PROP_POS_FRAMES));
    skip_frames(begin_skip);

    for (size_t i = det_begin; i <= det_end; i += skip_frames(skip_count) + 1) {
        cv::Mat frame;
        *m_video_ptr >> frame;
        if (frame.empty()) {
            Log.error(i, "frame is empty");
            callback(AsstMsg::SubTaskError, basic_info_with_what("DetectOperators"));
            return false;
        }

        cv::resize(frame, frame, cv::Size(), dectect_scale, dectect_scale, cv::INTER_AREA);

        int x_start = (frame.cols - 1280) / 2;
        int y_start = (frame.rows - 720) / 2;

        // 截取中心视口 (Rect 自动处理边界)
        cv::Mat standard_view = frame(cv::Rect(x_start, y_start, 1280, 720)).clone();

        BattlefieldDetector analyzer(standard_view);
        analyzer.set_object_of_interest({ .operators = true });
        auto result_opt = analyzer.analyze();

        DetectionResult cur_locations;
        auto tiles = m_normal_tile_info | std::views::values;
        for (const auto& box : result_opt->operators) {
            Rect rect = box.rect.move(det_box_move);

            auto iter = std::ranges::find_if(tiles, [&](const TilePack::TileInfo& t) { return rect.include(t.pos); });
            if (iter == tiles.end()) {
                Log.warn(i, __FUNCTION__, "no pos", box.rect.to_string(), rect);
                continue;
            }
            cur_locations.emplace((*iter).loc);
        }
        oper_det_samping[std::move(cur_locations)] += 1;

        clip.random_frames.emplace_back(frame); // for classify_direction
    }

    /* 取众数 */
    auto oper_det_iter = std::ranges::max_element(oper_det_samping, [&](const auto& lhs, const auto& rhs) {
        return lhs.second < rhs.second;
    });
    if (oper_det_iter == oper_det_samping.end()) {
        Log.error(__FUNCTION__, "oper_det_samping is empty");
        callback(AsstMsg::SubTaskError, basic_info_with_what("DetectOperators"));
        return false;
    }

    for (const Point& loc : oper_det_iter->first) {
        clip.battlefield.emplace(loc, BattlefieldOper {});
    }

    callback(AsstMsg::SubTaskCompleted, basic_info_with_what("DetectOperators"));
    return true;
}

bool asst::CombatRecordRecognitionTask::classify_direction(ClipInfo& clip, ClipInfo* pre_clip_ptr)
{
    LogTraceFunction;

    if (!pre_clip_ptr) {
        Log.info("first clip, skip");
        callback(AsstMsg::SubTaskCompleted, basic_info_with_what("ClassifyDirection"));
        return true;
    }

    std::vector<Point> newcomer;
    for (const Point& loc : clip.battlefield | std::views::keys) {
        if (pre_clip_ptr->battlefield.contains(loc)) {
            continue;
        }
        newcomer.emplace_back(loc);
    }
    if (newcomer.empty()) {
        return true;
    }
    callback(AsstMsg::SubTaskStart, basic_info_with_what("ClassifyDirection"));

    /* classify direction */
    using Raw = BattlefieldClassifier::DeployDirectionResult::Raw;
    constexpr size_t ClsSize = BattlefieldClassifier::DeployDirectionResult::ClsSize;
    std::unordered_map<Point, Raw> dir_cls_sampling;

    for (const cv::Mat& frame : clip.random_frames) {
        BattlefieldClassifier analyzer(frame);
        analyzer.set_object_of_interest({ .skill_ready = false, .deploy_direction = true });
        for (const auto& loc : newcomer) {
            analyzer.set_base_point(m_normal_tile_info.at(loc).pos);
            auto result_opt = analyzer.analyze();
            for (size_t i = 0; i < ClsSize; ++i) {
                dir_cls_sampling[loc][i] += result_opt->deploy_direction.raw[i];
            }
        }
    }

    for (const auto& [loc, sampling] : dir_cls_sampling) {
        auto class_id = std::max_element(sampling.begin(), sampling.end()) - sampling.begin();

        clip.battlefield[loc].direction = static_cast<battle::DeployDirection>(class_id);
        clip.battlefield[loc].new_here = true;
    }
    callback(AsstMsg::SubTaskCompleted, basic_info_with_what("ClassifyDirection"));
    return true;
}

bool asst::CombatRecordRecognitionTask::process_changes(ClipInfo& clip, ClipInfo* pre_clip_ptr)
{
    LogTraceFunction;
    std::ignore = m_video_ptr;
    std::ignore = clip;

    if (!pre_clip_ptr) {
        Log.info("first clip, skip");
        return true;
    }

    json::object condition = analyze_action_condition(clip, pre_clip_ptr);

    auto& actions_json = m_copilot_json["actions"].as_array();

    if (pre_clip_ptr->deployment.size() > clip.deployment.size() ||
        pre_clip_ptr->battlefield.size() < clip.battlefield.size()) {
        // 部署
        std::vector<std::string> deployed;
        ananlyze_deployment_names(clip);
        ananlyze_deployment_names(*pre_clip_ptr);
        for (const auto& pre_oper : pre_clip_ptr->deployment) {
            auto iter =
                std::ranges::find_if(clip.deployment, [&](const auto& oper) { return oper.name == pre_oper.name; });
            if (iter != clip.deployment.end()) {
                continue;
            }
            deployed.emplace_back(pre_oper.name);
        }
        Log.info("deployed", deployed);

        if (deployed.empty()) {
            Log.warn("Unknown dployed, will use pre tails_page's name", pre_clip_ptr->ends_oper_name);
        }

        auto deployed_iter = deployed.begin();
        for (const auto& [loc, oper] : clip.battlefield) {
            if (!oper.new_here) {
                continue;
            }
            std::string name = deployed_iter == deployed.end() ? pre_clip_ptr->ends_oper_name : *(deployed_iter++);
            if (name.empty()) {
                name = "Unknown_EndsEmpty";
            }

            static const std::unordered_map<battle::DeployDirection, std::string> DirectionNames = {
                { battle::DeployDirection::Right, "Right" }, { battle::DeployDirection::Down, "Down" },
                { battle::DeployDirection::Left, "Left" },   { battle::DeployDirection::Up, "Up" },
                { battle::DeployDirection::None, "None" },
            };
            const std::string& direction = DirectionNames.at(oper.direction);

            json::object deploy_json = condition | json::object {
                { "type", "Deploy" },
                { "name", name }, // 这里正常应该只有一个人，多了就只能抽奖了（
                { "location", json::array { loc.x, loc.y } },
                { "direction", direction },
            };
            Log.info("deploy json", deploy_json.to_string());
            actions_json.emplace_back(std::move(deploy_json));

            m_operator_locations.insert_or_assign(name, loc);
            m_location_operators.insert_or_assign(loc, name);
        }
    }
    else if (
        pre_clip_ptr->deployment.size() < clip.deployment.size() ||
        pre_clip_ptr->battlefield.size() > clip.battlefield.size()) {
        // 撤退
        for (const auto& [pre_loc, pre_oper] : pre_clip_ptr->battlefield) {
            if (clip.battlefield.contains(pre_loc)) {
                continue;
            }

            std::string name = m_location_operators[pre_loc];
            json::object retreat_json = condition | json::object {
                { "type", "Retreat" },
                { "location", json::array { pre_loc.x, pre_loc.y } },
                { "name", name },
            };
            Log.info("retreat json", retreat_json.to_string());
            actions_json.emplace_back(std::move(retreat_json));

            m_location_operators.erase(pre_loc);
            m_operator_locations.erase(name);
        }
    }
    else {
        Log.warn(
            "Unknown changes, deployment:",
            pre_clip_ptr->deployment.size(),
            clip.deployment.size(),
            "battlefield:",
            pre_clip_ptr->battlefield.size(),
            clip.battlefield.size());
    }

    return true;
}

void asst::CombatRecordRecognitionTask::ananlyze_deployment_names(ClipInfo& clip)
{
    LogTraceFunction;

    for (auto& oper : clip.deployment) {
        if (!oper.name.empty()) {
            continue;
        }
        BestMatcher avatar_analyzer(oper.avatar);
        static const double threshold = Task.get<MatchTaskInfo>("BattleAvatarDataForVideo")->templ_thresholds.front();
        avatar_analyzer.set_method(MatchMethod::Ccoeff);
        avatar_analyzer.set_threshold(threshold);
        // static const double drone_threshold = Task.get<MatchTaskInfo>("BattleDroneAvatarData")->templ_threshold;
        // avatar_analyzer.set_threshold(oper.role == battle::Role::Drone ? drone_threshold : threshold);

        for (const auto& [name, avatar] : m_all_avatars) {
            std::unordered_set<battle::Role> roles = { BattleData.get_role(name) };
            if (name == "阿米娅") {
                roles.emplace(battle::Role::Warrior);
            }
            if (roles.contains(oper.role)) {
                avatar_analyzer.append_templ(name, avatar);
            }
        }
        bool analyzed = avatar_analyzer.analyze().has_value();
        if (analyzed) {
            oper.name = avatar_analyzer.get_result().templ_info.name;
        }
        else {
            oper.name = "UnknownDeployment";
        }
    }
}

json::object asst::CombatRecordRecognitionTask::analyze_action_condition(ClipInfo& clip, ClipInfo* pre_clip_ptr)
{
    LogTraceFunction;

    if (m_frame_kills.empty()) {
        return {};
    }
    size_t target_frame_index = pre_clip_ptr->end_frame_index;
    size_t kill_frame_index = 0;
    int kills = 0;
    for (auto iter = m_frame_kills.begin() + 1; iter != m_frame_kills.end(); ++iter) {
        const auto& [cur_index, _] = *iter;
        if (cur_index < target_frame_index) {
            continue;
        }
        std::tie(kill_frame_index, kills) = *(iter - 1);
        break;
    }
    if (!kill_frame_index) {
        return {};
    }

    json::object condition {
        { "kills", kills },
    };

    BattlefieldMatcher analyzer(pre_clip_ptr->end_frame); // 开始执行这次操作的画面
    analyzer.set_object_of_interest({ .costs = true });
    auto end_result_opt = analyzer.analyze();
    if (!end_result_opt || end_result_opt->costs.status != BattlefieldMatcher::MatchStatus::Success) {
        m_pre_action_costs = -1;
        return condition;
    }
    int start_costs = end_result_opt->costs.value;
    int cost_changes = start_costs - m_pre_action_costs;
    if (m_pre_action_costs >= 0 && cost_changes > 0) {
        condition.emplace("cost_changes", cost_changes);
    }

    analyzer.set_image(clip.start_frame); // 这次操作执行完了的画面
    auto start_result_opt = analyzer.analyze();
    m_pre_action_costs = (start_result_opt && end_result_opt->costs.status == BattlefieldMatcher::MatchStatus::Success)
                             ? end_result_opt->costs.value
                             : start_costs;

    return condition;
}

size_t asst::CombatRecordRecognitionTask::skip_frames(size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        m_video_ptr->grab();
    }
    return count;
}

std::string asst::CombatRecordRecognitionTask::analyze_detail_page_oper_name(const cv::Mat& frame)
{
    const auto& replace_task = Task.get<OcrTaskInfo>("CharsNameOcrReplace");

    RegionOCRer preproc_analyzer(frame);
    preproc_analyzer.set_task_info("BattleOperName");
    preproc_analyzer.set_replace(replace_task->replace_map, replace_task->replace_full);
    auto preproc_result_opt = preproc_analyzer.analyze();

    if (preproc_result_opt && !BattleData.is_name_invalid(preproc_result_opt->text)) {
        return preproc_result_opt->text;
    }

    Log.warn("ocr with preprocess got a invalid name, try to use detect model");
    OCRer det_analyzer(frame);
    det_analyzer.set_task_info("BattleOperName");
    det_analyzer.set_replace(replace_task->replace_map, replace_task->replace_full);
    auto det_result_opt = det_analyzer.analyze();
    if (!det_result_opt) {
        return {};
    }
    sort_by_score_(*det_result_opt);
    const auto& det_name = det_result_opt->front().text;

    return BattleData.is_name_invalid(det_name) ? std::string() : det_name;
}

/*什么奇技淫巧艹*/
cv::Mat asst::CombatRecordRecognitionTask::get_stitched_720p(const cv::Mat& frame)
{
    int raw_w = frame.cols;
    int raw_h = frame.rows;
    double target_ratio = 1280.0 / 720.0;

    cv::Mat stitched = cv::Mat::zeros(720, 1280, frame.type());

    if (current_ratio > target_ratio) {
        // 宽屏锁定高度为 720 按照 1/4 和 3/4 处对称裁切原则
        double scale_h = 720.0 / raw_h;
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(), scale_h, scale_h, cv::INTER_AREA);

        int new_w = resized.cols;

        // 计算需要挖掉的总宽度，以及每个切口需要挖掉的宽度
        int redundant_w = new_w - 1280;
        int cut_w = redundant_w / 2;

        int q1 = new_w / 4;     // 左 1/4 处
        int q3 = new_w * 3 / 4; // 右 3/4 处

        // 左边保留区：从 0 到 q1 的切口起点
        int left_w = q1 - cut_w / 2;
        cv::Mat left_part = resized(cv::Rect(0, 0, left_w, 575)).clone();

        // 中间保留区：从 q1 切口终点，到 q3 切口起点
        int center_start = q1 + cut_w / 2;
        int center_w = (q3 - cut_w / 2) - center_start;
        cv::Mat center_part = resized(cv::Rect(center_start, 0, center_w, 575)).clone();

        // 右边保留区：从 q3 切口终点，到画面最右侧
        int right_start = q3 + cut_w / 2;
        int right_w = new_w - right_start;
        cv::Mat right_part =
            resized(cv::Rect(right_start, 0, right_w, 575) & cv::Rect(0, 0, resized.cols, 575)).clone();

        // 底部保留区：需要压缩(这里不需要了，但是懒得改了)
        int bottom_start = 575;
        int bottom_h = 720 - bottom_start;
        cv::Mat bottom_part = resized(cv::Rect(0, bottom_start, new_w, bottom_h)).clone();
        cv::resize(bottom_part, bottom_part, cv::Size(1280, bottom_h), 0, 0, cv::INTER_AREA);

        // 4. 严丝合缝地贴到 1280x720 的画布上
        int current_x = 0;
        left_part.copyTo(stitched(cv::Rect(current_x, 0, left_w, 575)));

        current_x += left_w;
        center_part.copyTo(stitched(cv::Rect(current_x, 0, center_w, 575)));

        current_x += center_w;
        int remaining_space = stitched.cols - current_x;
        int final_w = std::min(right_part.cols, remaining_space);
        right_part(cv::Rect(0, 0, final_w, 575)).copyTo(stitched(cv::Rect(current_x, 0, final_w, 575)));

        bottom_part.copyTo(stitched(cv::Rect(0, 575, 1280, bottom_h)));
    }
    else {
        // 窄屏锁定宽度 1280 保留顶部和底部 UI
        double scale_w = 1280.0 / raw_w;
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(), scale_w, scale_w, cv::INTER_AREA);

        int new_h = resized.rows;

        cv::Mat top_360 = resized(cv::Rect(0, 0, 1280, 360));
        cv::Mat bottom_360 = resized(cv::Rect(0, new_h - 360, 1280, 360));

        top_360.copyTo(stitched(cv::Rect(0, 0, 1280, 360)));
        bottom_360.copyTo(stitched(cv::Rect(0, 360, 1280, 360)));
    }
    // 返回前让我先笑会hhhhhhh

    return stitched;
}
