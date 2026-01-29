import java.io.File;
import java.io.IOException;
//public class manifestToImage {
//    public static void main(String[] args) throws IOException {
//        String picPath = "E:\\new_data\\mal\\mal_xml";
//        File file1 = new File(picPath);
//        File[] filePathLists1 = file1.listFiles();
//        System.out.println(filePathLists1.length);
//        for (int i = 1; i < filePathLists1.length+1; i++) {
//            File test = new File("E:\\new_data\\mal\\mal_xml\\"+i+".txt");
//            if(!test.exists()){
//                System.out.println("没有文件");
//            } else if (test.length() == 0){
//                System.out.println(i + "  no txt Color.");
//            } else {
//                hexToImage h = new hexToImage();
//                String s = h.readFileContent(picPath+"\\"+i+".txt");
//                byte[] bytes = h.hexToByteArray(s);
//                if (bytes.length / 600 <= 0){
//                    System.out.println("data is small, can be ignored");
//                } else {
//                    h.rgbBytesToJpg(bytes, 384, bytes.length / (384 * 3), "E:\\new_data\\mal\\mal_xml_images\\"+i+".jpg");
//                }
//            }
//        }
//    }
//
//}
import java.io.File;
import java.io.IOException;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

//public class manifestToImage {
//    public static void main(String[] args) throws IOException {
//        String txtPath = "E:\\raw_datas\\datas-time\\googleplay\\leg_xml\\2024";
//        File file1 = new File(txtPath);
//        File[] filePathLists1 = file1.listFiles();
//
//        if (filePathLists1 != null) {
//            Arrays.stream(filePathLists1)
//                    .filter(file -> file.getName().endsWith(".txt")) // 过滤 .txt 文件
//                    .forEach(file -> {
//                        String fileNameWithoutExtension = file.getName().replace(".txt", "");
//                        File test = new File(txtPath + File.separator + fileNameWithoutExtension + ".txt");
//                        if (!test.exists()) {
//                            System.out.println("没有文件: " + fileNameWithoutExtension + ".txt");
//                        } else if (test.length() == 0) {
//                            System.out.println(fileNameWithoutExtension + " no txt Color.");
//                        } else {
//                            hexToImage h = new hexToImage();
//                            try {
//                                String s = h.readFileContent(txtPath + File.separator + fileNameWithoutExtension + ".txt");
//                                System.out.println(s);
//
//                                byte[] bytes = h.hexToByteArray(s);
//                                if (bytes.length / 1536 <= 0) {
//                                    System.out.println("数据太小，可以忽略");
//                                } else {
//                                    h.rgbBytesToJpg(bytes, 512, bytes.length / 1536, "E:\\raw_datas\\datas-time\\googleplay\\leg_xml_image\\2024" + File.separator + "Leg." + fileNameWithoutExtension + ".jpg");
//                                }
//                            } catch (IOException e) {
//                                e.printStackTrace();
//                            }
//                        }
//                    });
//        } else {
//            System.out.println("没有找到文件");
//        }
//    }
//}
//
//
//
//import java.io.File;
//import java.io.IOException;
//
//public class manifestToImage {
//    public static void main(String[] args) throws IOException {
//        String picPath = "E:\\raw_datas\\datas-time\\googleplay\\leg_xml\\2024";  // 存放原始 .txt 文件的目录
//        File file1 = new File(picPath);
//        File[] filePathLists1 = file1.listFiles();
//
//        if (filePathLists1 != null) {
//            System.out.println("文件数量: " + filePathLists1.length);
//
//            for (int i = 0; i < filePathLists1.length; i++) {
//                // 获取每个文件的路径
//                File test = filePathLists1[i];
//
//                // 检查文件是否存在且非空
//                if (!test.exists()) {
//                    System.out.println(test.getName() + " 文件不存在");
//                } else if (test.length() == 0) {
//                    System.out.println(test.getName() + " 文件为空");
//                } else {
//                    // 使用原文件名生成输出图像文件
//                    hexToImage h = new hexToImage();
//                    String s = h.readFileContent(test.getAbsolutePath());
//                    byte[] bytes = h.hexToByteArray(s);
//
//                    // 检查数据是否足够大
//                    if (bytes.length / 1536 <= 0) {
//                        System.out.println("数据太小，跳过: " + test.getName());
//                    } else {
//                        // 使用原文件名生成 .jpg 文件并保存
//                        String outputImagePath = "E:\\raw_datas\\datas-time\\googleplay\\leg_xml_image\\2024\\" + test.getName().replace(".txt", ".jpg");
//                        h.rgbBytesToJpg(bytes, 512, bytes.length / (512 * 3), outputImagePath);
//                        System.out.println("转换完成: " + test.getName() + " -> " + outputImagePath);
//                    }
//                }
//            }
//        } else {
//            System.out.println("文件夹为空或路径不正确");
//        }
//    }
//}





//import javax.imageio.ImageIO;
//import java.awt.image.BufferedImage;
//import java.io.*;
//
//public class manifestToImage {
//    public static void main(String[] args) throws IOException {
//        String picPath = "E:\\new_data\\mal\\mal_xml"; // 输入文件夹路径
//        File file1 = new File(picPath);
//        File[] filePathLists1 = file1.listFiles();
//        System.out.println(filePathLists1.length);
//
//        for (int i = 1; i < filePathLists1.length + 1; i++) {
//            File test = new File(picPath + "\\" + i + ".txt");
//            if (!test.exists()) {
//                System.out.println("没有文件: " + i + ".txt");
//            } else if (test.length() == 0) {
//                System.out.println(i + "  no txt Color.");
//            } else {
//                String hexString = readHexStringFromFile(test); // 读取十六进制字符串
//                if (hexString == null || hexString.isEmpty()) {
//                    System.out.println(i + " contains invalid or empty hex data.");
//                    continue;
//                }
//
//                // 转换十六进制字符串为字节数组
//                byte[] bytes = hexStringToByteArray(hexString);
//                if (bytes.length / 600 <= 0) {
//                    System.out.println("data is small, can be ignored");
//                } else {
//                    rgbBytesToJpg(bytes, 384, bytes.length / (384 * 3), "E:\\new_data\\mal\\mal_xml_images\\" + i + ".jpg");
//                    System.out.println("Image saved: " + i + ".jpg");
//                }
//            }
//        }
//    }
//
//    // 从文件中读取内容并转换为十六进制字符串
//    public static String readHexStringFromFile(File file) throws IOException {
//        StringBuilder hexString = new StringBuilder();
//        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
//            String line;
//            while ((line = reader.readLine()) != null) {
//                line = line.trim(); // 去掉行首尾空格
//                System.out.println("Read line: " + line); // 打印读取的内容
//                if (!line.isEmpty()) {
//                    // 将每个字符转换为十六进制
//                    for (char ch : line.toCharArray()) {
//                        hexString.append(String.format("%02x", (int) ch));
//                    }
//                }
//            }
//        }
//        return hexString.toString();
//    }
//
//    // 将十六进制字符串转换为字节数组
//    public static byte[] hexStringToByteArray(String hex) {
//        int len = hex.length();
//        byte[] data = new byte[len / 2];
//        for (int i = 0; i < len; i += 2) {
//            data[i / 2] = (byte) ((Character.digit(hex.charAt(i), 16) << 4)
//                    + Character.digit(hex.charAt(i + 1), 16));
//        }
//        return data;
//    }
//
//    // 将RGB字节数组保存为JPEG图像
//    public static void rgbBytesToJpg(byte[] rgb, int width, int height, String path) throws IOException {
//        BufferedImage bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
//        bufferedImage.setRGB(0, 0, width, height, rgb24ToPixel(rgb, width, height), 0, width);
//        File file = new File(path);
//        ImageIO.write(bufferedImage, "jpg", file);
//    }
//
//    // 将字节数组的RGB颜色值转换为像素值
//    private static int[] rgb24ToPixel(byte[] rgb24, int width, int height) {
//        int[] pix = new int[rgb24.length / 3];
//        for (int i = 0; i < height; i++) {
//            for (int j = 0; j < width; j++) {
//                int idx = width * i + j;
//                int rgbIdx = idx * 3;
//                int red = rgb24[rgbIdx] & 0xFF;
//                int green = rgb24[rgbIdx + 1] & 0xFF;
//                int blue = rgb24[rgbIdx + 2] & 0xFF;
//                int color = (red << 16) | (green << 8) | blue;
//                pix[idx] = color;
//            }
//        }
//        return pix;
//    }
//}
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

import java.util.Arrays;


//把文本内容转成字节序列，再把字节序列按 RGB 三字节=1像素 排布到固定宽度的二维像素网格，生成一张图片。
public class manifestToImage {
    public static void main(String[] args) throws IOException {
        String picPath = "E:\\raw_datas\\datas-time\\googleplay_obf\\mal_xml\\2024"; // 输入文件夹路径
        File file1 = new File(picPath);
        File[] filePathLists1 = file1.listFiles();
        if (filePathLists1 != null) {
            System.out.println(filePathLists1.length);
            Arrays.stream(filePathLists1)  // 使用 Stream 来遍历文件
                    .filter(file -> file.getName().endsWith(".txt")) // 过滤 txt 文件
                    .forEach(test -> {
                        try {
                            if (test.length() == 0) {
                                System.out.println(test.getName() + "  no txt Color.");
                            } else {
                                String hexString = readHexStringFromFile(test); // 读取十六进制字符串
                                if (hexString == null || hexString.isEmpty()) {
                                    System.out.println(test.getName() + " contains invalid or empty hex data.");
                                } else {
                                    // 转换十六进制字符串为字节数组
                                    byte[] bytes = hexStringToByteArray(hexString);
                                    if (bytes.length / 384 <= 0) {
                                        System.out.println("data is small, can be ignored");
                                    } else {
                                        rgbBytesToJpg(bytes, 128, bytes.length / (128 * 3), "E:\\raw_datas\\datas-time\\googleplay_obf\\mal_xml_image\\2024\\" + "Mal." + test.getName().replace(".txt", ".jpg"));
                                        System.out.println("Image saved: " + test.getName());
                                    }
                                }
                            }
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    });
        }
    }

//     从文件中读取内容并转换为十六进制字符串
    public static String readHexStringFromFile(File file) throws IOException {
        StringBuilder hexString = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim(); // 去掉行首尾空格
                System.out.println("Read line: " + line); // 打印读取的内容
                if (!line.isEmpty()) {
                    // 将每个字符转换为十六进制
                    for (char ch : line.toCharArray()) {
                        hexString.append(String.format("%02x", (int) ch));
                    }
                }
            }
        }
        return hexString.toString();
    }



    // 将十六进制字符串转换为字节数组
//    public static byte[] hexStringToByteArray(String hex) {
//        int len = hex.length();
//        byte[] data = new byte[len / 2];
//        for (int i = 0; i < len; i += 2) {
//            data[i / 2] = (byte) ((Character.digit(hex.charAt(i), 16) << 4)
//                    + Character.digit(hex.charAt(i + 1), 16));
//        }
//        return data;
//    }
    // 将十六进制字符串转换为字节数组
    public static byte[] hexStringToByteArray(String hex) {
        int len = hex.length();

        // 如果十六进制字符串的长度为奇数，前面加 '0'，使长度变为偶数
        if (len % 2 != 0) {
            hex = "0" + hex;
            len++;
        }

        byte[] data = new byte[len / 2];
        for (int i = 0; i < len; i += 2) {
            // 每两个十六进制字符转换为一个字节
            data[i / 2] = (byte) ((Character.digit(hex.charAt(i), 16) << 4)
                    + Character.digit(hex.charAt(i + 1), 16));
        }
        return data;
    }


    // 将RGB字节数组保存为JPEG图像
    public static void rgbBytesToJpg(byte[] rgb, int width, int height, String path) throws IOException {
        BufferedImage bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        bufferedImage.setRGB(0, 0, width, height, rgb24ToPixel(rgb, width, height), 0, width);
        File file = new File(path);
        ImageIO.write(bufferedImage, "jpg", file);
    }

    // 将字节数组的RGB颜色值转换为像素值
    private static int[] rgb24ToPixel(byte[] rgb24, int width, int height) {
        // 每 3 个字节表示一个像素（R,G,B），所以像素个数 = 字节数 / 3
        int[] pix = new int[rgb24.length / 3];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int idx = width * i + j;
                int rgbIdx = idx * 3;
                int red = rgb24[rgbIdx] & 0xFF;
                int green = rgb24[rgbIdx + 1] & 0xFF;
                int blue = rgb24[rgbIdx + 2] & 0xFF;
                int color = (red << 16) | (green << 8) | blue;
                pix[idx] = color;
            }
        }
        return pix;
    }
}
