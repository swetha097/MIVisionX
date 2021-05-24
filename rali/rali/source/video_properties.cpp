#include "video_properties.h"
#include <cmath>

std::vector<unsigned> open_video_context(const char *video_file_path)
{
    std::vector<unsigned> video_prop;
    AVFormatContext *pFormatCtx = NULL;
    AVCodecContext *pCodecCtx = NULL;
    int videoStream = -1;
    unsigned int i = 0;
    float frame_rate;
    // open video file
    // std::cerr << "The video file path : " << video_file_path << "\n";
    int ret = avformat_open_input(&pFormatCtx, video_file_path, NULL, NULL);
    if (ret != 0)
    {
        std::cerr << "\nUnable to open video file:" << video_file_path << "\n";
        exit(0);
    }

    // Retrieve stream information
    ret = avformat_find_stream_info(pFormatCtx, NULL);
    assert(ret >= 0);

    for (i = 0; i < pFormatCtx->nb_streams; i++)
    {
        if (pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO && videoStream < 0)
        {
            videoStream = i;
        }
    } // end for i
    assert(videoStream != -1);

    // Get a pointer to the codec context for the video stream
    pCodecCtx = pFormatCtx->streams[videoStream]->codec;
    assert(pCodecCtx != NULL);
    //std::cerr<<"\n width:: "<<pCodecCtx->width;
    //std::cerr<<"\n height:: "<<pCodecCtx->height;
    frame_rate = (float)pFormatCtx->streams[videoStream]->avg_frame_rate.num / pFormatCtx->streams[videoStream]->avg_frame_rate.den;
    video_prop.push_back(pCodecCtx->width);
    video_prop.push_back(pCodecCtx->height);
    video_prop.push_back(pFormatCtx->streams[videoStream]->nb_frames);
    video_prop.push_back(round(frame_rate));
    avcodec_close(pCodecCtx);
    avformat_close_input(&pFormatCtx);
    return video_prop;
}

video_properties get_video_properties_from_txt_file(const char *file_path, bool enable_timestamps)
{
    std::ifstream text_file(file_path);

    if (text_file.good()) {
        //_text_file.open(path.c_str(), std::ifstream::in);
        video_properties video_props;
        std::vector<unsigned> props;
        std::string line;
        int label, max_width, max_height;
        int start, end;
        float start_time, end_time;
        int video_count = 0;
        std::string video_file_name;
        while (std::getline(text_file, line)) {
            start = end = max_width = max_height = 0;
            std::istringstream line_ss(line);
            if (!(line_ss >> video_file_name >> label))
                continue;
            props = open_video_context(video_file_name.c_str());
            if( props[0] >= max_width ) {    
                if (max_width != 0)           
                    std::cerr << "[WARN] The given video files are of different resolution\n";
                max_width = props[0];
            }
            if ( props[1] >= max_height ) {
                if (max_height != 0)
                    std::cerr << "[WARN] The given video files are of different resolution\n";
                max_height = props[1];
            }
            if(enable_timestamps) {
                if (line_ss >> start_time) {
                    if (line_ss >> end_time) {
                        if (start_time >= end_time) {
                            std::cerr << "[WRN] Start and end time/frame are not satisfying the condition, skipping the file " << video_file_name << "\n";
                            continue;
                        }
                        std::cerr << start_time << " : " << end_time << "\n";
                        std::cerr << "Frame rate : "  << props[3];
                        start = start_time * props[3];
                        end = end_time * props[3];
                        end = end ?end : props[2];
                        std::cerr << start<< " : " << end<< "\n";
                    }
                }
                video_props.start_end_timestamps.push_back(std::make_tuple(start_time, end_time)); 
            }
            else {
                if (line_ss >> start) {
                    if (line_ss >> end) {
                        if (start >= end) {
                            std::cerr << "[WRN] Start and end time/frame are the same, skipping the file " << video_file_name << "\n";
                            continue;
                        }
                        end = end ?end : props[2];
                    }
                }
            }
            if (end > props[2])
                THROW("The given frame numbers in txt file exceeds the maximum frames in the video" + video_file_name) 
            
            video_file_name = std::to_string(video_count) + "#" + video_file_name;
            video_props.video_file_names.push_back(video_file_name);
            video_props.labels.push_back(label);
            video_props.start_end_frame_num.push_back(std::make_tuple(start, end));
            video_props.frames_count.push_back(end - start);
            video_count++;
        }
        video_props.width = max_width;
        video_props.height = max_height;
        video_props.videos_count = video_count;
        return video_props;
    }
    else {
        THROW("Can't open the metadata file at " + std::string(file_path))
    }
}

video_properties find_video_properties(const char *source_path, bool enable_timestamps )
{
    // based on assumption that user can give single video file or path to folder containing
    // multiple video files.
    // check for videos in the path  is of same resolution. If not throw error and exit.

    DIR *_sub_dir;
    struct dirent *_entity;
    std::string vid_file_path;
    video_properties props;
    std::vector<unsigned> video_prop;
    // video_prop.resize(4);
    unsigned max_width = 0, max_height = 0;
    {
    std::string _full_path = source_path;
    filesys::path pathObj(_full_path);

    if (filesys::exists(pathObj) && filesys::is_regular_file(pathObj)) // Single file as input
    {
        if (pathObj.has_extension() && pathObj.extension().string() == ".txt")
        {
            // Fetch the extension from path object and return
            props = get_video_properties_from_txt_file(source_path, enable_timestamps);
        }
        else
        {
            video_prop = open_video_context(source_path);
            props.width = video_prop[0];
            props.height = video_prop[1];
            props.videos_count = 1;
            props.frames_count.push_back(video_prop[2]);
            props.start_end_frame_num.push_back(std::make_tuple(0, (int)video_prop[2]));
            vid_file_path = std::to_string(0) +  "#" + _full_path;
            props.video_file_names.push_back(vid_file_path);
        }
    }
    else if (filesys::exists(pathObj) && filesys::is_directory(pathObj))
    {
        //subfolder_reading(source_path, props);

        std::vector<std::string> video_files;
        unsigned video_count = 0;

        std::string _folder_path = source_path;
        if ((_sub_dir = opendir(_folder_path.c_str())) == nullptr)
            THROW("ERROR: Failed opening the directory at " + _folder_path);

        std::vector<std::string> entry_name_list;
        //std::string _full_path = _folder_path;

        while ((_entity = readdir(_sub_dir)) != nullptr)
        {
            std::string entry_name(_entity->d_name);
            if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0)
                continue;
            entry_name_list.push_back(entry_name);
        }
        closedir(_sub_dir);
        std::sort(entry_name_list.begin(), entry_name_list.end());

        for (unsigned dir_count = 0; dir_count < entry_name_list.size(); ++dir_count)
        {
            std::string subfolder_path = _folder_path + "/" + entry_name_list[dir_count];
            // std::cerr << "\nSubfodlerfile/ path :" << subfolder_path.c_str();
            filesys::path pathObj(subfolder_path);
            if (filesys::exists(pathObj) && filesys::is_regular_file(pathObj))
            {
                video_prop = open_video_context(subfolder_path.c_str());
                max_width = video_prop[0];
                max_height = video_prop[1];
                //props.width = video_prop[0];
                //props.height = video_prop[1];
                props.frames_count.push_back(video_prop[2]);
                vid_file_path = std::to_string(video_count) +  "#" + subfolder_path;
                props.video_file_names.push_back(vid_file_path);
                // props.video_file_names.push_back(subfolder_path);
                props.start_end_frame_num.push_back(std::make_tuple(0, (int)video_prop[2]));
                video_count++;
            }
            else if (filesys::exists(pathObj) && filesys::is_directory(pathObj))
            {
                std::string _full_path = subfolder_path;
                if ((_sub_dir = opendir(_full_path.c_str())) == nullptr)
                    THROW("VideoReader ERROR: Failed opening the directory at " + source_path);

                while ((_entity = readdir(_sub_dir)) != nullptr)
                {
                    std::string entry_name(_entity->d_name);
                    if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0)
                        continue;
                    video_files.push_back(entry_name);
                    // std::cerr << "\n  Inside video files : " << entry_name;
                    //++video_count;
                }
                closedir(_sub_dir);
                std::sort(video_files.begin(), video_files.end());
                for (unsigned i = 0; i < video_files.size(); i++)
                {
                    std::string file_path = _full_path;
                    file_path.append("/");
                    file_path.append(video_files[i]);
                    _full_path = file_path;

                    // std::cerr << "\n Props file name : " << _full_path;

                    video_prop = open_video_context(_full_path.c_str());
                    if (video_prop[0] > max_width || video_prop[1] > max_height && (max_width != 0 && max_height != 0))
                    {
                        max_width = video_prop[0];
                        std::cerr << "[WARN] The given video files are of different resolution\n";
                    }
                    if (video_prop[1] > max_height)
                        max_height = video_prop[1];
                    vid_file_path = std::to_string(video_count) +  "#" + _full_path;
                    props.video_file_names.push_back(vid_file_path);
                    // props.video_file_names.push_back(_full_path);
                    props.frames_count.push_back(video_prop[2]);
                    props.start_end_frame_num.push_back(std::make_tuple(0, (int)video_prop[2]));
                    video_count++;
                    _full_path = subfolder_path;
                }
                //exit(0);
                video_files.clear();
            }
        }
        props.videos_count = video_count;
        props.width = max_width;
        props.height = max_height;
    }
    }
    return props;
}