package scs.util.tools;
 
import java.util.Random;
 
/**
 * 随机字符串生成器类
 * @author yanan
 *
 */
public class RandomString {
	public static enum Mode {
		ALPHA, ALPHANUMERIC, NUMERIC
	}

	private static Random rand=new Random();
    public static final String SOURCES ="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890";
    
	public static String generateRandomString(int length, Mode mode){
		StringBuffer buffer = new StringBuffer();
		String characters = "";
		switch (mode) {

		case ALPHA:
			characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
			break;

		case ALPHANUMERIC:
			characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";
			break;

		case NUMERIC:
			characters = "1234567890";
			break;
		}

		int charactersLength = characters.length();

		for (int i = 0; i < length; i++) {
			double index = Math.random() * charactersLength;
			buffer.append(characters.charAt((int) index));
		}
		return buffer.toString();
	}
 
    /**
     * Generate a random string.
     *
     * @param random the random number generator.
     * @param characters the characters for generating string.
     * @param length the length of the generated string.
     * @return
     */
    public static String generateString(int length) {
        char[] text = new char[length];
        for (int i = 0; i < length; i++) {
            text[i] = SOURCES.charAt(rand.nextInt(62));
        }
        return new String(text);
    }
}