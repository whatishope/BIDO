//import java.awt.Color;
//import java.awt.Graphics;
//import java.awt.image.BufferedImage;
//import javax.imageio.ImageIO;
//import java.io.*;
//import java.nio.ByteBuffer;
//import java.nio.ByteOrder;
//import java.util.HashMap;
//
//public class DexRGBImageConverter {
//
//    private static final int IMAGE_WIDTH = 384; // 每行像素数
//
//    // 解析 DEX 头部信息，获取索引部分大小和偏移量
//    public static HashMap<String, Integer> getDexInfo(String dexPath) throws IOException {
//        HashMap<String, Integer> info = new HashMap<>();
//        FileInputStream fis = new FileInputStream(dexPath);
//        byte[] header = new byte[112]; // 读取 DEX 头部（前 112 字节）
//        fis.read(header);
//        fis.close();
//
//        // 解析 DEX 二进制数据（小端序）
//        ByteBuffer buffer = ByteBuffer.wrap(header);
//        buffer.order(ByteOrder.LITTLE_ENDIAN);
//
//        // 获取各个索引部分的大小和偏移量
//        info.put("stringIdSize", buffer.getInt(0x38));
//        info.put("stringIdOffset", buffer.getInt(0x3C));
//        info.put("typeIdSize", buffer.getInt(0x40));
//        info.put("typeIdOffset", buffer.getInt(0x44));
//        info.put("protoIdSize", buffer.getInt(0x48));
//        info.put("protoIdOffset", buffer.getInt(0x4C));
//        info.put("fieldIdSize", buffer.getInt(0x50));
//        info.put("fieldIdOffset", buffer.getInt(0x54));
//        info.put("methodIdSize", buffer.getInt(0x58));
//        info.put("methodIdOffset", buffer.getInt(0x5C));
//        info.put("classIdSize", buffer.getInt(0x60));
//        info.put("classIdOffset", buffer.getInt(0x64));
//
//        return info;
//    }
//
//    // 读取 DEX 文件的某个部分
//    public static byte[] extractSection(String dexPath, int offset, int size) throws IOException {
//        FileInputStream fis = new FileInputStream(dexPath);
//        fis.skip(offset);  // 跳转到指定偏移
//        byte[] buffer = new byte[size];
//        fis.read(buffer);
//        fis.close();
//        return buffer;
//    }
//
//    // 生成 RGB 图片
//    public static void generateRGBImage(String dexPath) throws IOException {
//        HashMap<String, Integer> dexInfo = getDexInfo(dexPath);
//
//        // 计算各部分大小
//        int stringSize = dexInfo.get("stringIdSize") * 4;
//        int typeSize = dexInfo.get("typeIdSize") * 4;
//        int protoSize = dexInfo.get("protoIdSize") * 12;
//        int fieldSize = dexInfo.get("fieldIdSize") * 8;
//        int methodSize = dexInfo.get("methodIdSize") * 8;
//        int classSize = dexInfo.get("classIdSize") * 32;
//
//        // 读取各个部分数据
//        byte[] stringData = extractSection(dexPath, dexInfo.get("stringIdOffset"), stringSize);
//        byte[] typeData = extractSection(dexPath, dexInfo.get("typeIdOffset"), typeSize);
//        byte[] protoData = extractSection(dexPath, dexInfo.get("protoIdOffset"), protoSize);
//        byte[] fieldData = extractSection(dexPath, dexInfo.get("fieldIdOffset"), fieldSize);
//        byte[] methodData = extractSection(dexPath, dexInfo.get("methodIdOffset"), methodSize);
//        byte[] classData = extractSection(dexPath, dexInfo.get("classIdOffset"), classSize);
//
//        // 合并所有数据
//        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
//        outputStream.write(stringData);
//        outputStream.write(typeData);
//        outputStream.write(protoData);
//        outputStream.write(fieldData);
//        outputStream.write(methodData);
//        outputStream.write(classData);
//        byte[] allData = outputStream.toByteArray();
//
//        // 计算图片高度
//        int totalPixels = allData.length / 3; // 每 3 个字节 1 像素
//        int height = (int) Math.ceil((double) totalPixels / IMAGE_WIDTH);
//
//        // 创建 RGB 图片
//        BufferedImage image = new BufferedImage(IMAGE_WIDTH, height, BufferedImage.TYPE_INT_RGB);
//        Graphics g = image.getGraphics();
//
//        // 遍历数据并绘制像素
//        for (int i = 0; i < totalPixels; i++) {
//            int index = i * 3;
//            int r = (index < allData.length) ? (allData[index] & 0xFF) : 0;
//            int gVal = (index + 1 < allData.length) ? (allData[index + 1] & 0xFF) : 0;
//            int b = (index + 2 < allData.length) ? (allData[index + 2] & 0xFF) : 0;
//
//            // 计算像素位置
//            int x = i % IMAGE_WIDTH;
//            int y = i / IMAGE_WIDTH;
//            if (y < height) {
//                image.setRGB(x, y, new Color(r, gVal, b).getRGB());
//            }
//        }
//
//        // 保存图片
//        ImageIO.write(image, "png", new File("dex_rgb_image.png"));
//        System.out.println("RGB 图片已保存: dex_rgb_image.png");
//    }
//
//    public static void main(String[] args) throws IOException {
//        String dexFile = "E:\\new_datas0-5\\leg\\leg_dex\\1\\classes.dex"; // 你的 DEX 文件路径
//        generateRGBImage(dexFile);
//    }
//}



//import java.awt.Color;
//import java.awt.Graphics;
//import java.awt.image.BufferedImage;
//import javax.imageio.ImageIO;
//import java.io.*;
//import java.nio.ByteBuffer;
//import java.nio.ByteOrder;
//import java.util.HashMap;

//public class DexRGBImageConverter {
//
//    private static final int IMAGE_WIDTH = 384; // 每行像素数
//
//    // 解析 DEX 头部信息，获取索引部分大小和偏移量
//    public static HashMap<String, Integer> getDexInfo(String dexPath) throws IOException {
//        HashMap<String, Integer> info = new HashMap<>();
//        FileInputStream fis = new FileInputStream(dexPath);
//        byte[] header = new byte[112]; // 读取 DEX 头部（前 112 字节）
//        fis.read(header);
//        fis.close();
//
//        // 解析 DEX 二进制数据（小端序）
//        ByteBuffer buffer = ByteBuffer.wrap(header);
//        buffer.order(ByteOrder.LITTLE_ENDIAN);
//
//        // 获取各个索引部分的大小和偏移量
//        info.put("stringIdSize", buffer.getInt(0x38));
//        info.put("stringIdOffset", buffer.getInt(0x3C));
//        info.put("typeIdSize", buffer.getInt(0x40));
//        info.put("typeIdOffset", buffer.getInt(0x44));
//        info.put("protoIdSize", buffer.getInt(0x48));
//        info.put("protoIdOffset", buffer.getInt(0x4C));
//        info.put("fieldIdSize", buffer.getInt(0x50));
//        info.put("fieldIdOffset", buffer.getInt(0x54));
//        info.put("methodIdSize", buffer.getInt(0x58));
//        info.put("methodIdOffset", buffer.getInt(0x5C));
//        info.put("classIdSize", buffer.getInt(0x60));
//        info.put("classIdOffset", buffer.getInt(0x64));
//
//        return info;
//    }
//
//    // 读取 DEX 文件的某个部分
//    public static byte[] extractSection(String dexPath, int offset, int size) throws IOException {
//        FileInputStream fis = new FileInputStream(dexPath);
//        fis.skip(offset);  // 跳转到指定偏移
//        byte[] buffer = new byte[size];
//        fis.read(buffer);
//        fis.close();
//        return buffer;
//    }
//
//    // 生成 RGB 图片并保存为 JPG
//    public static void generateRGBImage(String dexPath) throws IOException {
//        HashMap<String, Integer> dexInfo = getDexInfo(dexPath);
//
//        // 计算各部分大小
//        int stringSize = dexInfo.get("stringIdSize") * 4;
//        int typeSize = dexInfo.get("typeIdSize") * 4;
//        int protoSize = dexInfo.get("protoIdSize") * 12;
//        int fieldSize = dexInfo.get("fieldIdSize") * 8;
//        int methodSize = dexInfo.get("methodIdSize") * 8;
//        int classSize = dexInfo.get("classIdSize") * 32;
//
//        // 输出各部分的大小
//        System.out.println(" Dex 索引表大小（单位：字节）：");
//        System.out.println(" string_ids (字符串索引表)    : " + stringSize + " bytes");
//        System.out.println(" type_ids (类型索引表)        : " + typeSize + " bytes");
//        System.out.println(" proto_ids (方法原型索引表)   : " + protoSize + " bytes");
//        System.out.println(" field_ids (字段索引表)       : " + fieldSize + " bytes");
//        System.out.println(" method_ids (方法索引表)      : " + methodSize + " bytes");
//        System.out.println(" class_defs (类定义索引表)    : " + classSize + " bytes");
//
//        // 读取各个部分数据
//        byte[] stringData = extractSection(dexPath, dexInfo.get("stringIdOffset"), stringSize);
//        byte[] typeData = extractSection(dexPath, dexInfo.get("typeIdOffset"), typeSize);
//        byte[] protoData = extractSection(dexPath, dexInfo.get("protoIdOffset"), protoSize);
//        byte[] fieldData = extractSection(dexPath, dexInfo.get("fieldIdOffset"), fieldSize);
//        byte[] methodData = extractSection(dexPath, dexInfo.get("methodIdOffset"), methodSize);
//        byte[] classData = extractSection(dexPath, dexInfo.get("classIdOffset"), classSize);
//
//        // 合并所有数据
//        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
//        outputStream.write(stringData);
//        outputStream.write(typeData);
//        outputStream.write(protoData);
//        outputStream.write(fieldData);
//        outputStream.write(methodData);
//        outputStream.write(classData);
//        byte[] allData = outputStream.toByteArray();
//
//        // 计算图片高度
//        int totalPixels = allData.length / 3; // 每 3 个字节 1 像素
//        int height = (int) Math.ceil((double) totalPixels / IMAGE_WIDTH);
//
//        // 创建 RGB 图片
//        BufferedImage image = new BufferedImage(IMAGE_WIDTH, height, BufferedImage.TYPE_INT_RGB);
//        Graphics g = image.getGraphics();
//
//        // 遍历数据并绘制像素
//        for (int i = 0; i < totalPixels; i++) {
//            int index = i * 3;
//            int r = (index < allData.length) ? (allData[index] & 0xFF) : 0;
//            int gVal = (index + 1 < allData.length) ? (allData[index + 1] & 0xFF) : 0;
//            int b = (index + 2 < allData.length) ? (allData[index + 2] & 0xFF) : 0;
//
//            // 计算像素位置
//            int x = i % IMAGE_WIDTH;
//            int y = i / IMAGE_WIDTH;
//            if (y < height) {
//                image.setRGB(x, y, new Color(r, gVal, b).getRGB());
//            }
//        }
//
//        // 保存为 JPG 图片
//        File output = new File("dex_rgb_image.jpg");
//        ImageIO.write(image, "jpg", output);
//        System.out.println("RGB 图片已保存: " + output.getAbsolutePath());
//    }
//
//    public static void main(String[] args) throws IOException {
//        String dexFile = "E:\\new_datas0-5\\leg\\leg_dex\\1\\classes.dex"; // 你的 DEX 文件路径
//        generateRGBImage(dexFile);
//    }
//}


import java.awt.Color;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;

//对每个 APK 的 DEX 文件夹，把其中所有 *.dex 的索引区（string/type/proto/field/method/class 六类 ID 表）抽出来、按类别分别拼接，
// 再合并成一串二进制数据，按 RGB 三字节=1像素 映射生成一张 JPG 图（宽固定为 512），保存到输出目录。

public class DexRGBImageConverter {

    private static final int IMAGE_WIDTH = 512; // 图片的宽度

    /*
    * 读取 DEX 头部前 112 字节（dex header 固定长度 0x70）。
    用小端序解析头中各索引表的条目数与偏移，并乘以每条目的固定字节数得到“字节大小”：
    string_ids：每项 4B → count*4
    type_ids：每项 4B → count*4
    proto_ids：每项 12B → count*1
    field_ids：每项 8B → count*8
    method_ids：每项 8B → count*8
    class_defs：每项 32B → count*32
    注意：这里拿到的是索引表本身的原始二进制，不包含它们引用的 data 区（如 string_data、code_item 等）。
    * */

    // 解析 DEX 头部信息，获取索引部分大小和偏移量
    public static HashMap<String, Integer> getDexInfo(String dexPath) throws IOException {
        HashMap<String, Integer> info = new HashMap<>();
        FileInputStream fis = new FileInputStream(dexPath);
        byte[] header = new byte[112]; // 读取 DEX 头部
        fis.read(header);
        fis.close();

        // 小端序解析
        ByteBuffer buffer = ByteBuffer.wrap(header);
        buffer.order(ByteOrder.LITTLE_ENDIAN);

        // 读取索引部分大小和偏移量
        // string_ids：每项 4B，位置：size @ 0x38，off @ 0x3C
        info.put("stringIdSize", buffer.getInt(0x38) * 4); // 总字节数 = 项数 × 4
        info.put("stringIdOffset", buffer.getInt(0x3C));  // 起始偏移（字节）
        info.put("typeIdSize", buffer.getInt(0x40) * 4);
        info.put("typeIdOffset", buffer.getInt(0x44));
        info.put("protoIdSize", buffer.getInt(0x48) * 12);
        info.put("protoIdOffset", buffer.getInt(0x4C));
        info.put("fieldIdSize", buffer.getInt(0x50) * 8);
        info.put("fieldIdOffset", buffer.getInt(0x54));
        info.put("methodIdSize", buffer.getInt(0x58) * 8);
        info.put("methodIdOffset", buffer.getInt(0x5C));
        info.put("classIdSize", buffer.getInt(0x60) * 32);
        info.put("classIdOffset", buffer.getInt(0x64));

        return info;
    }

    // 读取 DEX 文件的某个部分
    public static byte[] extractSection(String dexPath, int offset, int size) throws IOException {
        FileInputStream fis = new FileInputStream(dexPath);
        fis.skip(offset);
        byte[] buffer = new byte[size];
        fis.read(buffer);
        fis.close();
        return buffer;
    }

    // 分别合并所有 DEX 文件的各索引部分
    public static byte[] mergeDexIndexes(String[] dexPaths) throws IOException {
        ByteArrayOutputStream stringData = new ByteArrayOutputStream();
        ByteArrayOutputStream typeData = new ByteArrayOutputStream();
        ByteArrayOutputStream protoData = new ByteArrayOutputStream();
        ByteArrayOutputStream fieldData = new ByteArrayOutputStream();
        ByteArrayOutputStream methodData = new ByteArrayOutputStream();
        ByteArrayOutputStream classData = new ByteArrayOutputStream();

        for (String dexPath : dexPaths) {
            HashMap<String, Integer> dexInfo = getDexInfo(dexPath);

            // 逐个合并索引部分
            stringData.write(extractSection(dexPath, dexInfo.get("stringIdOffset"), dexInfo.get("stringIdSize")));
            typeData.write(extractSection(dexPath, dexInfo.get("typeIdOffset"), dexInfo.get("typeIdSize")));
            protoData.write(extractSection(dexPath, dexInfo.get("protoIdOffset"), dexInfo.get("protoIdSize")));
            fieldData.write(extractSection(dexPath, dexInfo.get("fieldIdOffset"), dexInfo.get("fieldIdSize")));
            methodData.write(extractSection(dexPath, dexInfo.get("methodIdOffset"), dexInfo.get("methodIdSize")));
            classData.write(extractSection(dexPath, dexInfo.get("classIdOffset"), dexInfo.get("classIdSize")));
        }

        // 最终合并所有索引数据
        ByteArrayOutputStream finalData = new ByteArrayOutputStream();
        finalData.write(stringData.toByteArray());
        finalData.write(typeData.toByteArray());
        finalData.write(protoData.toByteArray());
        finalData.write(fieldData.toByteArray());
        finalData.write(methodData.toByteArray());
        finalData.write(classData.toByteArray());

        return finalData.toByteArray();
    }

    // 生成 RGB 图片
    public static void generateRGBImage(byte[] allData, String outputPath) throws IOException {
        int totalPixels = allData.length / 3;
        int height = (int) Math.ceil((double) totalPixels / IMAGE_WIDTH);

        BufferedImage image = new BufferedImage(IMAGE_WIDTH, height, BufferedImage.TYPE_INT_RGB);
        Graphics g = image.getGraphics();

        for (int i = 0; i < totalPixels; i++) {
            int index = i * 3;
            int r = (index < allData.length) ? (allData[index] & 0xFF) : 0;
            int gVal = (index + 1 < allData.length) ? (allData[index + 1] & 0xFF) : 0;
            int b = (index + 2 < allData.length) ? (allData[index + 2] & 0xFF) : 0;

            int x = i % IMAGE_WIDTH;
            int y = i / IMAGE_WIDTH;
            if (y < height) {
                image.setRGB(x, y, new Color(r, gVal, b).getRGB());
            }
        }

        File outputFile = new File(outputPath);
        ImageIO.write(image, "jpg", outputFile);
        System.out.println(" RGB 图片已保存: " + outputFile.getAbsolutePath());
    }

    // 处理一个子文件夹中的所有 DEX 文件
    public static void processDexFolder(String folderPath, String outputFolder) throws IOException {
        File folder = new File(folderPath);
        if (!folder.exists() || !folder.isDirectory()) {
            System.out.println(" 目录不存在: " + folderPath);
            return;
        }

        File[] dexFiles = folder.listFiles((dir, name) -> name.endsWith(".dex"));
        if (dexFiles == null || dexFiles.length == 0) {
            System.out.println(" 目录中没有 DEX 文件: " + folderPath);
            return;
        }

        String[] dexPaths = Arrays.stream(dexFiles).map(File::getAbsolutePath).toArray(String[]::new);
        byte[] mergedData = mergeDexIndexes(dexPaths);

        String outputFilePath = outputFolder + File.separator + "Mal." + folder.getName() + ".jpg";
        generateRGBImage(mergedData, outputFilePath);
    }

    // 处理所有 APK 目录
    public static void processAllDexFolders(String rootFolder, String outputFolder) throws IOException {
        File rootDir = new File(rootFolder);
        if (!rootDir.exists() || !rootDir.isDirectory()) {
            System.out.println(" 目录不存在: " + rootFolder);
            return;
        }

        for (File subDir : rootDir.listFiles()) {
            if (subDir.isDirectory()) {
                processDexFolder(subDir.getAbsolutePath(), outputFolder);
            }
        }
        System.out.println(" 所有 APK 处理完成！");
    }

    public static void main(String[] args) throws IOException {
        String rootFolder = "E:\\raw_datas\\datas-time\\googleplay_obf\\mal_dex\\2024";
        String outputFolder = "E:\\raw_datas\\datas-time\\googleplay_obf\\mal_dex_image\\2024";

        processAllDexFolders(rootFolder, outputFolder);
    }
}
