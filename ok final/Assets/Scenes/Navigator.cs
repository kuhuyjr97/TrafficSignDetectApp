using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
public class Navigator : MonoBehaviour
{
    // Start is called before the first frame update
    public void test1()
    {
        SceneManager.LoadScene("test1");
    }
    public void home()
    {
        SceneManager.LoadScene("home");
    }
    public void home1()
    {
        SceneManager.LoadScene("home1");
    }
    public void test2()
    {
        SceneManager.LoadScene("test2");
    }
    public void test3()
    {
        SceneManager.LoadScene("test3");
    }
}
