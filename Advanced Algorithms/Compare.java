import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class Compare{
  private static final String FILE_PATH = "./GatewayDevice_saro.txt";
  private static final String COMPARED_FILE_PATH = "./GatewayDevice_we.txt";
  private static final String RESULT_FILE_PATH = "./result1.txt";
  public static void main(String[] args) {
    System.out.println("======Start Search!=======");
    long startTime = System.currentTimeMillis();
    // Read first file
    File file = new File(FILE_PATH);
    File comparedFile = new File(COMPARED_FILE_PATH);
    BufferedReader br = null;
    BufferedReader cbr = null;
    BufferedWriter rbw = null;
    try {
      br = new BufferedReader(new FileReader(file));
      cbr = new BufferedReader(new FileReader(comparedFile));
      cbr.mark(90000000);
      rbw = new BufferedWriter(new FileWriter(RESULT_FILE_PATH));
      String lineText = null;
      while ((lineText = br.readLine()) != null) {
        String searchText = lineText.trim();
        searchAndSignProcess(searchText, cbr, rbw);
      }
      long endTime = System.currentTimeMillis();
      System.out.println("======Process Over!=======");
      System.out.println("Time Spending:" + ((endTime - startTime) / 1000D) + "s");
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    } catch (IOException e) {
      e.printStackTrace();
    } finally {
      if (br != null) {
        try {
          br.close();
        } catch (IOException e) {
          e.printStackTrace();
        } finally {
          if (cbr != null && rbw != null) {
            try {
              cbr.close();
              rbw.close();
            } catch (IOException e) {
              e.printStackTrace();
            }
          }
        }
      }
    }
  }
  public static void searchAndSignProcess(String searchText, BufferedReader comparedReader, BufferedWriter rbw)
      throws IOException {
    String lineStr = "-\n";
    if (searchText == null) {
      return;
    }
    if ("".equals(searchText)) {
      rbw.write(lineStr);
      return;
    }
    String lineText = null;
    int lineNum = 1;
    while ((lineText = comparedReader.readLine()) != null) {
      String comparedLine = lineText.trim();
      if (searchText.equals(comparedLine)) {
        lineStr = "###=Equal:" + lineNum + "=###\n";
        break;
      }
      lineNum++;
    }
    rbw.write(lineStr);
    comparedReader.reset();
  }
}
