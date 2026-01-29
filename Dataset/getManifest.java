//import net.dongliu.apk.parser.ApkFile;
//import net.dongliu.apk.parser.exception.ParserException;
//
//import java.io.File;
//import java.io.FileWriter;
//import java.io.IOException;
//
//public class getManifest1 {
//    public static void main(String[] args) throws IOException {
//        String legPath = "E:\\new_data\\leg\\leg_apk1\\";
//        String malPath = "E:\\new_data\\mal\\mal_apk1\\";
//        String malOutPath = "E:\\new_data\\mal\\mal_xml\\";
//        String legOutPath = "E:\\new_data\\leg\\leg_xml\\";
//        for (int i = 1; i <= 3386; i++) {
//            String path = legPath + i + ".apk";
//            String out = legOutPath + i + ".txt";
//            if (manifestGet(path, out)){
//                System.out.println(i+"  complete");
//            } else {
//                File file = new File(out);
//                file.createNewFile();
//            }
//        }
//    }
//
//    public static boolean manifestGet(String path, String outpath) throws IOException {
//        try {
//            ApkFile apkFile = new ApkFile(new File(path));
//            String manifestXml = apkFile.getManifestXml();
//            FileWriter fwriter = null;
//            fwriter = new FileWriter(outpath, false);
//            fwriter.write(manifestXml);
//            fwriter.flush();
//            fwriter.close();
//            return true;
//        } catch (ParserException e) {
//            System.out.println("no manifest!");
//            return false;
//        }
//    }
//}
import net.dongliu.apk.parser.ApkFile;
import net.dongliu.apk.parser.exception.ParserException;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;


//遍历一批 APK → 把每个 APK 全量解压到指定目录 → 再把除 AndroidManifest.xml 之外的内容都删掉（并删子目录），最终只保留每个 APK 的 Manifest 文件。
/*
* 用 apk-parser 库把 APK 里的 二进制 AXML（AndroidManifest.xml）解码成可读 XML 字符串；
    把结果写到 outPath\同名.txt；
    如果解析失败（ParserException），就建一个空的占位 .txt。
* */

public class getManifest {
    public static void main(String[] args) throws IOException {
        String legPath = "E:\\raw_datas\\datas-time\\googleplay_obf\\mal\\2024\\";
        String malPath = "E:\\new_data\\mal\\mal_apk1\\";
        String malOutPath = "E:\\new_data\\mal\\mal_xml\\";
        String legOutPath = "E:\\raw_datas\\datas-time\\googleplay_obf\\mal_xml\\2024\\";

        // 递归遍历目录中的 APK 文件
        File legDirectory = new File(legPath);
        processApkFiles(legDirectory, legOutPath);
    }

    public static void processApkFiles(File directory, String outPath) throws IOException {
        File[] files = directory.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.isDirectory()) {
                    // 递归处理子文件夹
                    processApkFiles(file, outPath);
                } else if (file.getName().endsWith(".apk")) {
                    // 处理 APK 文件
                    String out = outPath + file.getName().replace(".apk", ".txt");
                    if (manifestGet(file.getAbsolutePath(), out)) {
                        System.out.println(file.getName() + " complete");
                    } else {
                        new File(out).createNewFile();
                    }
                }
            }
        }
    }

    public static boolean manifestGet(String path, String outpath) throws IOException {
        try {
            ApkFile apkFile = new ApkFile(new File(path));
            String manifestXml = apkFile.getManifestXml();
            FileWriter fwriter = new FileWriter(outpath, false);
            fwriter.write(manifestXml);
            fwriter.flush();
            fwriter.close();
            return true;
        } catch (ParserException e) {
            System.out.println("no manifest in: " + path);
            return false;
        }
    }
}


